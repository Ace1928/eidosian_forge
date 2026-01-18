import math
from typing import List, Optional, Tuple
import torch
class _EmformerLayer(torch.nn.Module):
    """Emformer layer that constitutes Emformer.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        ffn_dim: (int): hidden layer dimension of feedforward network.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in feedforward network.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, segment_length: int, dropout: float=0.0, activation: str='relu', left_context_length: int=0, max_memory_size: int=0, weight_init_gain: Optional[float]=None, tanh_on_mem: bool=False, negative_inf: float=-100000000.0):
        super().__init__()
        self.attention = _EmformerAttention(input_dim=input_dim, num_heads=num_heads, dropout=dropout, weight_init_gain=weight_init_gain, tanh_on_mem=tanh_on_mem, negative_inf=negative_inf)
        self.dropout = torch.nn.Dropout(dropout)
        self.memory_op = torch.nn.AvgPool1d(kernel_size=segment_length, stride=segment_length, ceil_mode=True)
        activation_module = _get_activation_module(activation)
        self.pos_ff = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, ffn_dim), activation_module, torch.nn.Dropout(dropout), torch.nn.Linear(ffn_dim, input_dim), torch.nn.Dropout(dropout))
        self.layer_norm_input = torch.nn.LayerNorm(input_dim)
        self.layer_norm_output = torch.nn.LayerNorm(input_dim)
        self.left_context_length = left_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size
        self.input_dim = input_dim
        self.use_mem = max_memory_size > 0

    def _init_state(self, batch_size: int, device: Optional[torch.device]) -> List[torch.Tensor]:
        empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
        left_context_key = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        left_context_val = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
        return [empty_memory, left_context_key, left_context_val, past_length]

    def _unpack_state(self, state: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_length = state[3][0][0].item()
        past_left_context_length = min(self.left_context_length, past_length)
        past_mem_length = min(self.max_memory_size, math.ceil(past_length / self.segment_length))
        pre_mems = state[0][self.max_memory_size - past_mem_length:]
        lc_key = state[1][self.left_context_length - past_left_context_length:]
        lc_val = state[2][self.left_context_length - past_left_context_length:]
        return (pre_mems, lc_key, lc_val)

    def _pack_state(self, next_k: torch.Tensor, next_v: torch.Tensor, update_length: int, mems: torch.Tensor, state: List[torch.Tensor]) -> List[torch.Tensor]:
        new_k = torch.cat([state[1], next_k])
        new_v = torch.cat([state[2], next_v])
        state[0] = torch.cat([state[0], mems])[-self.max_memory_size:]
        state[1] = new_k[new_k.shape[0] - self.left_context_length:]
        state[2] = new_v[new_v.shape[0] - self.left_context_length:]
        state[3] = state[3] + update_length
        return state

    def _process_attention_output(self, rc_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor) -> torch.Tensor:
        result = self.dropout(rc_output) + torch.cat([right_context, utterance])
        result = self.pos_ff(result) + result
        result = self.layer_norm_output(result)
        return result

    def _apply_pre_attention_layer_norm(self, utterance: torch.Tensor, right_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_norm_input = self.layer_norm_input(torch.cat([right_context, utterance]))
        return (layer_norm_input[right_context.size(0):], layer_norm_input[:right_context.size(0)])

    def _apply_post_attention_ffn(self, rc_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rc_output = self._process_attention_output(rc_output, utterance, right_context)
        return (rc_output[right_context.size(0):], rc_output[:right_context.size(0)])

    def _apply_attention_forward(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, mems: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError('attention_mask must be not None when for_inference is False')
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m = self.attention(utterance=utterance, lengths=lengths, right_context=right_context, summary=summary, mems=mems, attention_mask=attention_mask)
        return (rc_output, next_m)

    def _apply_attention_infer(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, mems: torch.Tensor, state: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if state is None:
            state = self._init_state(utterance.size(1), device=utterance.device)
        pre_mems, lc_key, lc_val = self._unpack_state(state)
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m, next_k, next_v = self.attention.infer(utterance=utterance, lengths=lengths, right_context=right_context, summary=summary, mems=pre_mems, left_context_key=lc_key, left_context_val=lc_val)
        state = self._pack_state(next_k, next_v, utterance.size(0), mems, state)
        return (rc_output, next_m, state)

    def forward(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, mems: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    encoded utterance frames, with shape `(T, B, D)`.
                Tensor
                    updated right context frames, with shape `(R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        layer_norm_utterance, layer_norm_right_context = self._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems = self._apply_attention_forward(layer_norm_utterance, lengths, layer_norm_right_context, mems, attention_mask)
        output_utterance, output_right_context = self._apply_post_attention_ffn(rc_output, utterance, right_context)
        return (output_utterance, output_right_context, output_mems)

    @torch.jit.export
    def infer(self, utterance: torch.Tensor, lengths: torch.Tensor, right_context: torch.Tensor, state: Optional[List[torch.Tensor]], mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass for inference.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            state (List[torch.Tensor] or None): list of tensors representing layer internal state
                generated in preceding invocation of ``infer``.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.

        Returns:
            (Tensor, Tensor, List[torch.Tensor], Tensor):
                Tensor
                    encoded utterance frames, with shape `(T, B, D)`.
                Tensor
                    updated right context frames, with shape `(R, B, D)`.
                List[Tensor]
                    list of tensors representing layer internal state
                    generated in current invocation of ``infer``.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        layer_norm_utterance, layer_norm_right_context = self._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems, output_state = self._apply_attention_infer(layer_norm_utterance, lengths, layer_norm_right_context, mems, state)
        output_utterance, output_right_context = self._apply_post_attention_ffn(rc_output, utterance, right_context)
        return (output_utterance, output_right_context, output_state, output_mems)