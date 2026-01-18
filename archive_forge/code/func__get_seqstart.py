import math
from dataclasses import dataclass
from typing import (
import torch
@classmethod
def _get_seqstart(cls, seqlens: Iterable[int]) -> Tuple[int, int, List[int], torch.Tensor]:
    """
        Given sequence lengths, returns the min/max value and the sequence start
        positions (offsets), with first element being 0 (returned in list and Tensor).
        """
    assert not isinstance(seqlens, torch.Tensor)
    seqstart_py = [0]
    max_seqlen = -1
    min_seqlen = -1
    for seqlen in seqlens:
        min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
        max_seqlen = max(max_seqlen, seqlen)
        seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
    seqstart = torch.tensor(seqstart_py, dtype=torch.int32)
    return (min_seqlen, max_seqlen, seqstart_py, seqstart)