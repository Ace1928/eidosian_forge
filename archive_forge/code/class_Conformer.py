from typing import Optional, Tuple
import torch
class Conformer(torch.nn.Module):
    """Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, depthwise_conv_kernel_size: int, dropout: float=0.0, use_group_norm: bool=False, convolution_first: bool=False):
        super().__init__()
        self.conformer_layers = torch.nn.ModuleList([ConformerLayer(input_dim, ffn_dim, num_heads, depthwise_conv_kernel_size, dropout=dropout, use_group_norm=use_group_norm, convolution_first=convolution_first) for _ in range(num_layers)])

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return (x.transpose(0, 1), lengths)