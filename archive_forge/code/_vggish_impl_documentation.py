import math
import torch

        Args:
            input (torch.Tensor): waveform, with shape `(T,)`.
            sample_rate (int): sample rate of waveform in hertz.

        Returns:
            torch.Tensor: batch of examples to pass to VGGish, with shape `(n_example, 1, n_frame, 64)`.
        