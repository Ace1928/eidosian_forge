from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
@dataclass(init=False)
class PipeMessage:
    src: int
    dest: int
    queue_name: int
    args: Any
    tensors: Tensors
    tensor_shapes: List[torch.Size]
    tensor_dtypes: List[torch.dtype]
    tag: int = 0

    def __init__(self, src: int, dest: int, queue_name: int, args: Any=None, tensors: Optional[Tensors]=None, tensor_count: int=0):
        self.src = src
        self.dest = dest
        self.queue_name = queue_name
        self.args = args
        self.tensors = tensors or tuple()
        self.tensor_shapes = []
        self.tensor_dtypes = []
        global MessageGeneration
        self.tag = MessageGeneration
        if tensors is None:
            MessageGeneration += tensor_count
        else:
            MessageGeneration += len(self.tensors)