import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class _GappySeqInfo(_SeqLenInfo):
    """
    (Internal)  Represents the division of a dimension into blocks which are
    anywhere. Each just has a start and a length. The final start is the total
    length of the dimension.

    For example, to represent a dimension of length 14 like follows with
    three occupied lengths of
    6, 3 and 1, use `from_seqlens_padded([0, 7, 12, 14], [6, 3, 1])`.

    The layout along the dimension is

     0 ─►  block 0
           block 0
           block 0
           block 0
     4 ─►  block 0
           block 0
           <space>
           block 1
     8 ─►  block 1
           block 1
           <space>
           <space>
     12 ─► block 2
           <space>

    The members will be:
        max_seqlen: 6
        min_seqlen: 1
        seqstart_py: [0, 7, 12, 14]
        seqstart: torch.IntTensor([0, 7, 12, 14])
        seqlen_py: [6, 3 1]
        seqlen: torch.IntTensor([6, 3, 1])
    """
    seqlen: torch.Tensor
    seqlen_py: Sequence[int]

    def __post_init__(self) -> None:
        assert len(self.seqstart_py) == len(self.seqlen_py) + 1

    def to(self, device: torch.device) -> None:
        self.seqlen = self.seqlen.to(device, non_blocking=True)
        super().to(device)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield (start, start + length)

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> '_SeqLenInfo':
        raise NotImplementedError()

    @classmethod
    def from_seqlens_gappy(cls, seqstarts: Sequence[int], seqlens: Sequence[int]) -> '_GappySeqInfo':
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = list(seqstarts)
        if len(seqlens) == 0:
            raise ValueError('No elements')
        if len(seqstarts) - len(seqlens) != 1:
            raise ValueError(f'len(seqstarts) {seqstarts} should be 1 + len(seqlens) {seqlens}')
        seqlen = torch.tensor(seqlens, dtype=torch.int32)
        return cls(seqlen=seqlen, seqlen_py=seqlens, max_seqlen=max(seqlens), min_seqlen=min(seqlens), seqstart=torch.tensor(seqstart_py, dtype=torch.int32), seqstart_py=seqstart_py)

    def split(self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]]=None) -> List[torch.Tensor]:
        raise NotImplementedError('_PaddedSeqLenInfo.split')