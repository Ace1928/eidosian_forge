import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(self, root: Union[Path, str], sample_rate: float, seq_duration: Optional[float]=None, source_augmentations: Optional[Callable]=None) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = 'Dataset ' + self.__class__.__name__
        body = ['Number of datapoints: {}'.format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [' ' * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ''