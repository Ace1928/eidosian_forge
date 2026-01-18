import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .vision import VisionDataset
class Flickr8k(VisionDataset):
    """`Flickr8k Entities <http://hockenmaier.cs.illinois.edu/8k-pictures.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root: str, ann_file: str, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        parser = Flickr8kParser(self.root)
        with open(self.ann_file) as fh:
            parser.feed(fh.read())
        self.annotations = parser.annotations
        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def __len__(self) -> int:
        return len(self.ids)