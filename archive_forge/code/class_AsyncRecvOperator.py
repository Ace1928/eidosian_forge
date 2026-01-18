from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import ProcessGroup
from fairscale.nn.model_parallel import get_pipeline_parallel_ranks
from .checkpoint import Checkpointing
from .messages import Transport
from .microbatch import Batch
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .types import EVENT_LOOP_QUEUE, PipeMessage, TensorOrTensors, Tensors
from .worker import Task
class AsyncRecvOperator(torch.autograd.Function):
    """Receive activations to the previous pipeline stage"""

    @staticmethod
    def forward(ctx, phony: Tensor, transport: Transport, message: PipeMessage, queue_name: int) -> Tensors:
        ctx.transport = transport
        ctx.index = message.args.microbatch_index
        ctx.queue_name = queue_name
        result = transport.recv_message_tensors(message)
        ctx.args = result.args

        def maybe_requires_grad(t: Tensor) -> Tensor:
            if t.dtype.is_floating_point:
                return t.requires_grad_()
            return t
        return tuple((maybe_requires_grad(r) for r in result.tensors))

    @staticmethod
    def backward(ctx, *grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        ranks = get_pipeline_parallel_ranks()
        this_rank = torch.distributed.get_rank()
        body = AsyncMessageBody(AsyncMessageType.Gradients, ctx.index, source=ctx.args.dest, dest=ctx.args.source, order=ctx.args.order - 1)
        ctx.transport.send_message(PipeMessage(this_rank, ranks[ctx.args.source.stage], queue_name=ctx.queue_name, args=body, tensors=tuple(grad)), sync=True)
        tail_ctx = getattr(ctx, 'tail_ctx', None)
        if tail_ctx:
            expected_gradients = tail_ctx.expected_gradients
            while expected_gradients > 0:
                message = ctx.transport.recv_message_header(ctx.queue_name)
                args: AsyncMessageBody = message.args
                assert args.message_type is AsyncMessageType.Gradients
                invocation = tail_ctx.invocations[args.order]
                expected_gradients -= tail_ctx.count_per_order[invocation.order]
                AsyncEventLoop.perform_backward_for_invocation(ctx.transport, message, tail_ctx.activations, invocation)
        return (None, None, None, None, None)