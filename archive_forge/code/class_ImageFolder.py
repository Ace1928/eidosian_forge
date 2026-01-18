import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from .vision import VisionDataset
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root: str, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, loader: Callable[[str], Any]=default_loader, is_valid_file: Optional[Callable[[str], bool]]=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples