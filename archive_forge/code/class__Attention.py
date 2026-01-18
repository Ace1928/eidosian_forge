import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
class _Attention(nn.Module):
    """Locally sensitive attention model.

    Args:
        attention_rnn_dim (int): Number of hidden units for RNN.
        encoder_embedding_dim (int): Number of embedding dimensions in the Encoder.
        attention_hidden_dim (int): Dimension of attention hidden representation.
        attention_location_n_filter (int): Number of filters for Attention model.
        attention_location_kernel_size (int): Kernel size for Attention model.
    """

    def __init__(self, attention_rnn_dim: int, encoder_embedding_dim: int, attention_hidden_dim: int, attention_location_n_filter: int, attention_location_kernel_size: int) -> None:
        super().__init__()
        self.query_layer = _get_linear_layer(attention_rnn_dim, attention_hidden_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = _get_linear_layer(encoder_embedding_dim, attention_hidden_dim, bias=False, w_init_gain='tanh')
        self.v = _get_linear_layer(attention_hidden_dim, 1, bias=False)
        self.location_layer = _LocationLayer(attention_location_n_filter, attention_location_kernel_size, attention_hidden_dim)
        self.score_mask_value = -float('inf')

    def _get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
        """Get the alignment vector.

        Args:
            query (Tensor): Decoder output with shape (n_batch, n_mels * n_frames_per_step).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, attention_hidden_dim).
            attention_weights_cat (Tensor): Cumulative and previous attention weights
                with shape (n_batch, 2, max of ``text_lengths``).

        Returns:
            alignment (Tensor): attention weights, it is a tensor with shape (batch, max of ``text_lengths``).
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        alignment = energies.squeeze(2)
        return alignment

    def forward(self, attention_hidden_state: Tensor, memory: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Pass the input through the Attention model.

        Args:
            attention_hidden_state (Tensor): Attention rnn last output with shape (n_batch, ``attention_rnn_dim``).
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
            attention_weights_cat (Tensor): Previous and cumulative attention weights
                with shape (n_batch, current_num_frames * 2, max of ``text_lengths``).
            mask (Tensor): Binary mask for padded data with shape (n_batch, current_num_frames).

        Returns:
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
        """
        alignment = self._get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        alignment = alignment.masked_fill(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return (attention_context, attention_weights)