import math
from dataclasses import dataclass
from typing import (
import torch
@dataclass
class BlockDiagonalPaddedKeysMask(AttentionBias):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`,
    except we support padding for k/v

    The keys and values are divided into blocks which are padded out to
    the same total length.
    For example, if there is space for 12 keys, for three blocks of
    max length 4, but we only want to use the first 2, 3 and 2
    of each block, use `kv_padding=4` and `kv_seqlens=[2, 3, 2]`.
    The queries are divided into blocks, without padding, of lengths given by
    q_seqlen.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area).
    """
    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo

    def materialize(self, shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu') -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
            raise ValueError('k shapes wrong')
        if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
            raise ValueError('q shapes wrong')
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals())):
            mask[q_start:q_end, k_start:k_end] = 0
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(cls, q_seqlen: Sequence[int], kv_padding: int, kv_seqlen: Sequence[int], causal_diagonal: Any=None) -> 'BlockDiagonalPaddedKeysMask':
        """Creates a :attr:`BlockDiagonalPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            BlockDiagonalPaddedKeysMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (q_seqlen, kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    def make_paged(self, block_tables: torch.Tensor, page_size: int, paged_type: Type['PagedBlockDiagonalPaddedKeysMask']) -> AttentionBias:
        paged_bias = paged_type(q_seqinfo=self.q_seqinfo, k_seqinfo=self.k_seqinfo, block_tables=block_tables, page_size=page_size)
        paged_bias.k_seqinfo.padding = block_tables.shape[1] * page_size
        return paged_bias