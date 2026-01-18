from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
class RpcTransport(Transport):

    def send_message(self, message: PipeMessage, sync: bool=False, skip_header: bool=False) -> None:
        message.tensors = tuple((t.cpu() for t in message.tensors))
        assert self.worker_map
        name = self.worker_map[message.dest]
        if sync:
            torch.distributed.rpc.rpc_sync(name, rpc_push_queue, args=(message,))
        else:
            torch.distributed.rpc.rpc_async(name, rpc_push_queue, args=(message,))

    def recv_message_header(self, queue_name: int, nowait: bool=False) -> PipeMessage:
        queue = MessageQueues[queue_name]
        if nowait:
            result = queue.get_nowait()
        else:
            result = queue.get()
        result.tensors = to_input_device(result.tensors, self.input_device)
        return result

    def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
        message.tensors = to_input_device(message.tensors, self.input_device)
        return message

    def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
        """Receive a message with a known microbatch index, and handle out-of-order
        messages by placing them back on the queue"""
        queue = globals()['MessageQueues'][queue_name]
        out_of_order: List[PipeMessage] = []
        while True:
            message = self.recv_message(queue_name)
            got_index = message.args
            value = message.tensors
            if got_index == index:
                for b in out_of_order:
                    queue.put(b)
                return value
            else:
                out_of_order.append(message)