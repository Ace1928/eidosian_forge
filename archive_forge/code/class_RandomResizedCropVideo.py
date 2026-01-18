import numbers
import random
import warnings
from torchvision.transforms import RandomCrop, RandomResizedCrop
from . import _functional_video as F
class RandomResizedCropVideo(RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation_mode='bilinear'):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f'size should be tuple (height, width), instead got {size}')
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}, scale={self.scale}, ratio={self.ratio})'