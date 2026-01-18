from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from .data import DataConsumer
class MultiInputSequential(nn.Module):
    """A variation of nn.Sequential, that allows the first module in the sequence accepts
    multiple inputs. To be used internally by _split_module
    """

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, *inputs: Tuple[Tensor]) -> Tensor:
        input = self.modules_list[0](*inputs)
        for module in self.modules_list[1:]:
            input = module(input)
        return input