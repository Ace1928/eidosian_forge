from typing import Any, Callable, Optional, Tuple
import torch
from .. import transforms
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        