import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
class UpsampleNetwork(nn.Module):
    """Upscale the dimensions of a spectrogram.

    Args:
        upsample_scales: the list of upsample scales.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> upsamplenetwork = UpsampleNetwork(upsample_scales=[4, 4, 16])
        >>> input = torch.rand(10, 128, 10)  # a random spectrogram
        >>> output = upsamplenetwork(input)  # shape: (10, 128, 1536), (10, 128, 1536)
    """

    def __init__(self, upsample_scales: List[int], n_res_block: int=10, n_freq: int=128, n_hidden: int=128, n_output: int=128, kernel_size: int=5) -> None:
        super().__init__()
        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        self.total_scale: int = total_scale
        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        up_layers = []
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, scale * 2 + 1), padding=(0, scale), bias=False)
            torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1))
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward(self, specgram: Tensor) -> Tuple[Tensor, Tensor]:
        """Pass the input through the UpsampleNetwork layer.

        Args:
            specgram (Tensor): the input sequence to the UpsampleNetwork layer (n_batch, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale),
                          (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        where total_scale is the product of all elements in upsample_scales.
        """
        resnet_output = self.resnet(specgram).unsqueeze(1)
        resnet_output = self.resnet_stretch(resnet_output)
        resnet_output = resnet_output.squeeze(1)
        specgram = specgram.unsqueeze(1)
        upsampling_output = self.upsample_layers(specgram)
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent:-self.indent]
        return (upsampling_output, resnet_output)