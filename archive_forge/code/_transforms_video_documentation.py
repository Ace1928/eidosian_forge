import numbers
import random
import warnings
from torchvision.transforms import RandomCrop, RandomResizedCrop
from . import _functional_video as F

        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        