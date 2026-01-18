import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sized, Union, cast
import torch
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler, Sampler
from typing_extensions import Self, override
from lightning_fabric.utilities.distributed import _DatasetSamplerWrapper
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info
from pytorch_lightning.utilities.types import _SizedIterable
class UnrepeatedDistributedSamplerWrapper(UnrepeatedDistributedSampler):
    """Equivalent class to ``DistributedSamplerWrapper`` but for the ``UnrepeatedDistributedSampler``."""

    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())