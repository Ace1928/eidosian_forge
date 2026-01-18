from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import Tensor, nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from .data import DataConsumer
def _data_source_spec_to_data_source(self, spec: DataSourceSpec) -> DataSource:
    if isinstance(spec, int):
        return DataSource(None, spec)
    if isinstance(spec, RemoteModule):
        return DataSource(self._find_node(spec), 0)
    return DataSource(self._find_node(spec[0]), spec[1])