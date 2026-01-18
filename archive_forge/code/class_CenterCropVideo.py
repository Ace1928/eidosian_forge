import numbers
import random
import warnings
from torchvision.transforms import RandomCrop, RandomResizedCrop
from . import _functional_video as F
class CenterCropVideo:

    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.center_crop(clip, self.crop_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(crop_size={self.crop_size})'