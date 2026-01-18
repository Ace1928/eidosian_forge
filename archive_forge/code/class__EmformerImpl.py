import math
from typing import List, Optional, Tuple
import torch
class _EmformerImpl(torch.nn.Module):

    def __init__(self, emformer_layers: torch.nn.ModuleList, segment_length: int, left_context_length: int=0, right_context_length: int=0, max_memory_size: int=0):
        super().__init__()
        self.use_mem = max_memory_size > 0
        self.memory_op = torch.nn.AvgPool1d(kernel_size=segment_length, stride=segment_length, ceil_mode=True)
        self.emformer_layers = emformer_layers
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size

    def _gen_right_context(self, input: torch.Tensor) -> torch.Tensor:
        T = input.shape[0]
        num_segs = math.ceil((T - self.right_context_length) / self.segment_length)
        right_context_blocks = []
        for seg_idx in range(num_segs - 1):
            start = (seg_idx + 1) * self.segment_length
            end = start + self.right_context_length
            right_context_blocks.append(input[start:end])
        right_context_blocks.append(input[T - self.right_context_length:])
        return torch.cat(right_context_blocks)

    def _gen_attention_mask_col_widths(self, seg_idx: int, utterance_length: int) -> List[int]:
        num_segs = math.ceil(utterance_length / self.segment_length)
        rc = self.right_context_length
        lc = self.left_context_length
        rc_start = seg_idx * rc
        rc_end = rc_start + rc
        seg_start = max(seg_idx * self.segment_length - lc, 0)
        seg_end = min((seg_idx + 1) * self.segment_length, utterance_length)
        rc_length = self.right_context_length * num_segs
        if self.use_mem:
            m_start = max(seg_idx - self.max_memory_size, 0)
            mem_length = num_segs - 1
            col_widths = [m_start, seg_idx - m_start, mem_length - seg_idx, rc_start, rc, rc_length - rc_end, seg_start, seg_end - seg_start, utterance_length - seg_end]
        else:
            col_widths = [rc_start, rc, rc_length - rc_end, seg_start, seg_end - seg_start, utterance_length - seg_end]
        return col_widths

    def _gen_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
        utterance_length = input.size(0)
        num_segs = math.ceil(utterance_length / self.segment_length)
        rc_mask = []
        query_mask = []
        summary_mask = []
        if self.use_mem:
            num_cols = 9
            rc_q_cols_mask = [idx in [1, 4, 7] for idx in range(num_cols)]
            s_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
            masks_to_concat = [rc_mask, query_mask, summary_mask]
        else:
            num_cols = 6
            rc_q_cols_mask = [idx in [1, 4] for idx in range(num_cols)]
            s_cols_mask = None
            masks_to_concat = [rc_mask, query_mask]
        for seg_idx in range(num_segs):
            col_widths = self._gen_attention_mask_col_widths(seg_idx, utterance_length)
            rc_mask_block = _gen_attention_mask_block(col_widths, rc_q_cols_mask, self.right_context_length, input.device)
            rc_mask.append(rc_mask_block)
            query_mask_block = _gen_attention_mask_block(col_widths, rc_q_cols_mask, min(self.segment_length, utterance_length - seg_idx * self.segment_length), input.device)
            query_mask.append(query_mask_block)
            if s_cols_mask is not None:
                summary_mask_block = _gen_attention_mask_block(col_widths, s_cols_mask, 1, input.device)
                summary_mask.append(summary_mask_block)
        attention_mask = (1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])).to(torch.bool)
        return attention_mask

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training and non-streaming inference.

        B: batch size;
        T: max number of input frames in batch;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): utterance frames right-padded with right context frames, with
                shape `(B, T + right_context_length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid utterance frames for i-th batch element in ``input``.

        Returns:
            (Tensor, Tensor):
                Tensor
                    output frames, with shape `(B, T, D)`.
                Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        input = input.permute(1, 0, 2)
        right_context = self._gen_right_context(input)
        utterance = input[:input.size(0) - self.right_context_length]
        attention_mask = self._gen_attention_mask(utterance)
        mems = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:-1] if self.use_mem else torch.empty(0).to(dtype=input.dtype, device=input.device)
        output = utterance
        for layer in self.emformer_layers:
            output, right_context, mems = layer(output, lengths, right_context, mems, attention_mask)
        return (output.permute(1, 0, 2), lengths)

    @torch.jit.export
    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states: Optional[List[List[torch.Tensor]]]=None) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Forward pass for streaming inference.

        B: batch size;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): utterance frames right-padded with right context frames, with
                shape `(B, segment_length + right_context_length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            states (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation of ``infer``. (Default: ``None``)

        Returns:
            (Tensor, Tensor, List[List[Tensor]]):
                Tensor
                    output frames, with shape `(B, segment_length, D)`.
                Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
                List[List[Tensor]]
                    output states; list of lists of tensors representing internal state
                    generated in current invocation of ``infer``.
        """
        if input.size(1) != self.segment_length + self.right_context_length:
            raise ValueError(f'Per configured segment_length and right_context_length, expected size of {self.segment_length + self.right_context_length} for dimension 1 of input, but got {input.size(1)}.')
        input = input.permute(1, 0, 2)
        right_context_start_idx = input.size(0) - self.right_context_length
        right_context = input[right_context_start_idx:]
        utterance = input[:right_context_start_idx]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        mems = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1) if self.use_mem else torch.empty(0).to(dtype=input.dtype, device=input.device)
        output = utterance
        output_states: List[List[torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            output, right_context, output_state, mems = layer.infer(output, output_lengths, right_context, None if states is None else states[layer_idx], mems)
            output_states.append(output_state)
        return (output.permute(1, 0, 2), output_lengths, output_states)