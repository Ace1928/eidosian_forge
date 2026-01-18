import time
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from fairscale.nn.pipe.async_schedule import (
from fairscale.nn.pipe.checkpoint import Checkpointing
from fairscale.nn.pipe.messages import Transport
from fairscale.nn.pipe.microbatch import Batch
from fairscale.nn.pipe.types import (
from fairscale.nn.pipe.worker import Task
class AsyncAMPnetEventLoop:

    def __init__(self, partitions: List[ModuleWrapper], group: ProcessGroup, transport: Transport, min_update_interval: int, weight_prediction: bool, checkpoint_stop: int, input_device: Union[None, int, str, torch.device], chunks: int):
        self.partitions = partitions
        self.group = group
        self.transport = transport
        self.min_update_interval = min_update_interval
        self.weight_prediction = weight_prediction
        self.checkpoint_stop = checkpoint_stop
        self.input_device = input_device
        self.chunks = chunks

    def perform_optimizer_step(self, optimizer: Any, num_gradients: Any) -> Any:
        return optimizer is not None and (not self.weight_prediction and num_gradients % self.min_update_interval == 0) or (self.weight_prediction and num_gradients % self.chunks == 0)

    def async_send_inner(self, batch: Batch, index: int) -> Tuple[Batch, PipeMessage]:
        task = create_task_without_skip_trackers(self.checkpoint_stop, index, self.group.rank(), batch, self.partitions[0].module)
        result = task.compute()
        task.finalize(result)
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        body = AsyncMessageBody(AsyncMessageType.Activations, index, Location(this_rank, 0), Location(ranks[ranks.index(this_rank) + 1], 0), 0)
        message = PipeMessage(this_rank, ranks[ranks.index(this_rank) + 1], queue_name=EVENT_LOOP_ACTIVATIONS_QUEUE, args=body, tensors=tuple([*result]))
        return (result, message)

    def async_grad_inner(self, message: PipeMessage, activations: Dict[int, Batch]) -> None:
        args: AsyncMessageBody = message.args
        recvd_grads = self.transport.recv_message_tensors(message)
        batch = activations[args.microbatch_index]
        if len(recvd_grads.tensors) != len(batch):
            raise RuntimeError('different number of tensors and gradients')
        grads = []
        final_tensors = []
        for i, tensor in enumerate(batch):
            if tensor.requires_grad or getattr(tensor, 'grad_fn', None) is not None:
                grads.append(recvd_grads.tensors[i])
                final_tensors.append(tensor)
        torch.autograd.backward(final_tensors, grad_tensors=grads, retain_graph=True)
        del activations[args.microbatch_index]

    def get_batch_from_message(self, message: PipeMessage, queue_name: int) -> Batch:
        """Get the tensor(s) wrapped in a `Batch` from a `PipeMessage`, applying
        AsyncRecvOperator so we can intercept the backward pass"""
        microbatch_index = message.args.microbatch_index
        phony = torch.empty(0, device=self.transport.input_device, requires_grad=True)
        result = AsyncRecvOperator.apply(phony, self.transport, message, queue_name)
        if len(result) == 1:
            batch = Batch(result[0], microbatch_index)
        else:
            batch = Batch(result, microbatch_index)
        return batch

    def event_loop_head_across_minibatches(self, lm_dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer, transform_logger_object: Any) -> None:
        cur_rank = self.group.rank()
        N = len(get_pipeline_parallel_ranks())
        activations = dict()
        count = 0
        num_gradients = 0
        lm_iter = iter(lm_dataloader)
        while True:
            try:
                cur_batch = next(lm_iter)
                reqd_input = transform_logger_object.transform_input(cur_batch).to(self.input_device)
                batch = Batch(reqd_input, count)
                if self.weight_prediction:
                    optimizer.update_weight_using_future_predictions(cur_rank, N, count, self.chunks, forward=True)
                activations[count], message = self.async_send_inner(batch, count)
                self.transport.send_message(message, sync=True)
                count += 1
                if count == N - 1:
                    break
            except StopIteration:
                break
        while True:
            try:
                cur_batch = next(lm_iter)
                reqd_input = transform_logger_object.transform_input(cur_batch).to(self.input_device)
                batch = Batch(reqd_input, count)
                if self.weight_prediction:
                    optimizer.update_weight_using_future_predictions(cur_rank, N, count, self.chunks, forward=True)
                activations[count], forward_message = self.async_send_inner(batch, count)
                count += 1
                message = self.transport.recv_message_header(EVENT_LOOP_GRADIENTS_QUEUE)
                args: AsyncMessageBody = message.args
                assert args.message_type is AsyncMessageType.Gradients
                if self.weight_prediction:
                    optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
                self.async_grad_inner(message, activations)
                self.transport.send_message(forward_message, sync=True)
                num_gradients += 1
                if self.perform_optimizer_step(optimizer, num_gradients):
                    optimizer.step()
                    optimizer.zero_grad()
                    transform_logger_object.check_and_save_weights(num_gradients)
            except StopIteration:
                break
        remaining_items = len(activations)
        for _ in range(remaining_items):
            message = self.transport.recv_message_header(EVENT_LOOP_GRADIENTS_QUEUE)
            args = message.args
            assert args.message_type is AsyncMessageType.Gradients
            if self.weight_prediction:
                optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
            self.async_grad_inner(message, activations)
            num_gradients += 1
            if self.perform_optimizer_step(optimizer, num_gradients):
                optimizer.step()
                optimizer.zero_grad()
                transform_logger_object.check_and_save_weights(num_gradients)

    def event_loop_tail_across_minibatches(self, lm_dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer, transform_logger_object: Any) -> None:
        cur_rank = self.group.rank()
        N = len(get_pipeline_parallel_ranks())
        num_batches = len(lm_dataloader)
        lm_iter = enumerate(lm_dataloader)
        count = 0
        num_gradients = 0
        activations = dict()
        log_interval = 1
        word_counter = 0
        total_loss = 0
        while True:
            try:
                start_time = time.time()
                microbatch_index, cur_batch = next(lm_iter)
                reqd_target = transform_logger_object.transform_target(cur_batch).to(self.input_device)
                message = self.transport.recv_message_header(EVENT_LOOP_ACTIVATIONS_QUEUE)
                args: AsyncMessageBody = message.args
                assert args.microbatch_index == count
                batch = self.get_batch_from_message(message, EVENT_LOOP_GRADIENTS_QUEUE)
                if self.weight_prediction:
                    optimizer.update_weight_using_future_predictions(cur_rank, N, count, self.chunks, forward=True)
                task = create_task_without_skip_trackers(self.checkpoint_stop, args.microbatch_index, self.group.rank(), batch, self.partitions[0].module)
                output = task.compute()
                activations[args.microbatch_index] = output
                task.finalize(output)
                if self.weight_prediction:
                    optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
                output_tensor = transform_logger_object.transform_output_before_loss(output.tensor)
                loss = criterion(output_tensor, reqd_target)
                loss.backward()
                count += 1
                num_gradients += 1
                if self.perform_optimizer_step(optimizer, num_gradients):
                    optimizer.step()
                    optimizer.zero_grad()
                    transform_logger_object.check_and_save_weights(num_gradients)
                transform_logger_object.log_loss(cur_batch, loss, count)
                del loss
                del activations[args.microbatch_index]
            except StopIteration:
                break

    def event_loop_trunk_forward_helper(self, activations: Dict[int, Batch]) -> PipeMessage:
        message = self.transport.recv_message_header(EVENT_LOOP_ACTIVATIONS_QUEUE)
        args: AsyncMessageBody = message.args
        assert args.message_type is AsyncMessageType.Activations
        batch = self.get_batch_from_message(message, EVENT_LOOP_GRADIENTS_QUEUE)
        activations[args.microbatch_index], message = self.async_send_inner(batch, args.microbatch_index)
        return message

    def event_loop_trunk_backward_helper(self, activations: Dict[int, Batch]) -> None:
        message = self.transport.recv_message_header(EVENT_LOOP_GRADIENTS_QUEUE)
        args: AsyncMessageBody = message.args
        assert args.message_type is AsyncMessageType.Gradients
        self.async_grad_inner(message, activations)

    def event_loop_across_minibatches(self, lm_dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer, transform_logger_object: Any) -> None:
        activations: Dict[int, Batch] = dict()
        num_microbatch = len(lm_dataloader)
        num_activations = 0
        num_gradients = 0
        ranks = get_pipeline_parallel_ranks()
        N = len(ranks)
        cur_rank = torch.distributed.get_rank()
        n_warmup = ranks[-1] - cur_rank
        for _ in range(n_warmup):
            if self.weight_prediction:
                optimizer.update_weight_using_future_predictions(cur_rank, N, num_activations, self.chunks, forward=True)
            message = self.event_loop_trunk_forward_helper(activations)
            self.transport.send_message(message, sync=True)
            num_activations += 1
        while num_activations < num_microbatch:
            if self.weight_prediction:
                optimizer.update_weight_using_future_predictions(cur_rank, N, num_activations, self.chunks, forward=True)
            message = self.event_loop_trunk_forward_helper(activations)
            num_activations += 1
            if self.weight_prediction:
                optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
            self.event_loop_trunk_backward_helper(activations)
            num_gradients += 1
            if self.perform_optimizer_step(optimizer, num_gradients):
                optimizer.step()
                optimizer.zero_grad()
                transform_logger_object.check_and_save_weights(num_gradients)
            self.transport.send_message(message, sync=True)
        remaining = len(activations)
        for _ in range(remaining):
            if self.weight_prediction:
                optimizer.update_weight_using_future_predictions(cur_rank, N, num_gradients, self.chunks, forward=False)
            self.event_loop_trunk_backward_helper(activations)
            num_gradients += 1
            if self.perform_optimizer_step(optimizer, num_gradients):
                optimizer.step()
                optimizer.zero_grad()
                transform_logger_object.check_and_save_weights(num_gradients)