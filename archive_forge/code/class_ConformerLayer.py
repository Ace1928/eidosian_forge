from typing import Optional, Tuple
import torch
class ConformerLayer(torch.nn.Module):
    """Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(self, input_dim: int, ffn_dim: int, num_attention_heads: int, depthwise_conv_kernel_size: int, dropout: float=0.0, use_group_norm: bool=False, convolution_first: bool=False) -> None:
        super().__init__()
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(input_dim=input_dim, num_channels=input_dim, depthwise_kernel_size=depthwise_conv_kernel_size, dropout=dropout, bias=True, use_group_norm=use_group_norm)
        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual
        if self.convolution_first:
            x = self._apply_convolution(x)
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.self_attn_dropout(x)
        x = x + residual
        if not self.convolution_first:
            x = self._apply_convolution(x)
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        x = self.final_layer_norm(x)
        return x