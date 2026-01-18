import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    .. note::
        Before using this class, it is required to download ImageNet 2012 dataset from
        `here <https://image-net.org/challenges/LSVRC/2012/2012-downloads.php>`_ and
        place the files ``ILSVRC2012_devkit_t12.tar.gz`` and ``ILSVRC2012_img_train.tar``
        or ``ILSVRC2012_img_val.tar`` based on ``split`` in the root directory.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str='train', **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, 'split', ('train', 'val'))
        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]
        super().__init__(self.split_folder, **kwargs)
        self.root = root
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)
        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return 'Split: {split}'.format(**self.__dict__)