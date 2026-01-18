from abc import ABC
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue
from typing import Dict, List, Optional
import torch
from fairscale.internal.object import pyobject_to_tensor, tensor_to_pyobject
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from .types import MESSAGE_GENERATION_START, InputDevice, PipeMessage, Tensors
class SendRecvTransport(Transport):

    def send_message(self, message: PipeMessage, sync: bool=False, skip_header: bool=False) -> None:
        tensors = message.tensors
        message.tensors = tuple()
        torch.cuda.current_stream().synchronize()
        if not skip_header:
            message.tensor_shapes = [t.size() for t in tensors]
            message.tensor_dtypes = [t.dtype for t in tensors]
            torch.distributed.send(pyobject_to_tensor(message, MESSAGE_TENSOR_SIZE).cuda(), message.dest, tag=message.queue_name, group=get_pipeline_parallel_group())
        for index, t in enumerate(tensors):
            if t.device.type == 'cpu':
                t = t.cuda()
            torch.distributed.send(t.contiguous(), message.dest, tag=message.tag + index, group=get_pipeline_parallel_group())

    def recv_message_header(self, queue_name: int, nowait: bool=False) -> PipeMessage:
        if nowait:
            raise QueueEmpty
        tensor = torch.empty(MESSAGE_TENSOR_SIZE, dtype=torch.uint8, device=self.input_device)
        torch.cuda.current_stream().synchronize()
        torch.distributed.recv(tensor, src=None, tag=queue_name, group=get_pipeline_parallel_group())
        torch.cuda.current_stream().synchronize()
        return tensor_to_pyobject(tensor)

    def recv_message_tensors(self, message: PipeMessage) -> PipeMessage:
        torch.cuda.current_stream().synchronize()
        message_tensors = []
        for index, (shape, dtype) in enumerate(zip(message.tensor_shapes, message.tensor_dtypes)):
            t = torch.empty(*shape, dtype=dtype, device=self.input_device)
            torch.distributed.recv(t, message.src, tag=message.tag + index, group=get_pipeline_parallel_group())
            message_tensors.append(t)
        message.tensors = tuple(message_tensors)
        torch.cuda.current_stream().synchronize()
        return message

    def get_out_of_order(self, queue_name: int, index: int) -> Tensors:
        """Receive a message with a known microbatch index, and handle out-of-order
        messages by placing them back on the queue"""
        message = self.recv_message(queue_name)
        assert message.args == index
        return message.tensors