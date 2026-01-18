import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
class Caltech256(VisionDataset):
    """`Caltech 256 <https://data.caltech.edu/records/20087>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root: str, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, download: bool=False) -> None:
        super().__init__(os.path.join(root, 'caltech256'), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        self.categories = sorted(os.listdir(os.path.join(self.root, '256_ObjectCategories')))
        self.index: List[int] = []
        self.y = []
        for i, c in enumerate(self.categories):
            n = len([item for item in os.listdir(os.path.join(self.root, '256_ObjectCategories', c)) if item.endswith('.jpg')])
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root, '256_ObjectCategories', self.categories[self.y[index]], f'{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg'))
        target = self.y[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def _check_integrity(self) -> bool:
        return os.path.exists(os.path.join(self.root, '256_ObjectCategories'))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive('https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK', self.root, filename='256_ObjectCategories.tar', md5='67b4f42ca05d46448c6bb8ecd2220f6d')