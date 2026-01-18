import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
class LFWPairs(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold``. Defaults to ``10fold``.
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root: str, split: str='10fold', image_set: str='funneled', transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, download: bool=False) -> None:
        super().__init__(root, split, image_set, 'pairs', transform, target_transform, download)
        self.pair_names, self.data, self.targets = self._get_pairs(self.images_dir)

    def _get_pairs(self, images_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[int]]:
        pair_names, data, targets = ([], [], [])
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            if self.split == '10fold':
                n_folds, n_pairs = lines[0].split('\t')
                n_folds, n_pairs = (int(n_folds), int(n_pairs))
            else:
                n_folds, n_pairs = (1, int(lines[0]))
            s = 1
            for fold in range(n_folds):
                matched_pairs = [line.strip().split('\t') for line in lines[s:s + n_pairs]]
                unmatched_pairs = [line.strip().split('\t') for line in lines[s + n_pairs:s + 2 * n_pairs]]
                s += 2 * n_pairs
                for pair in matched_pairs:
                    img1, img2, same = (self._get_path(pair[0], pair[1]), self._get_path(pair[0], pair[2]), 1)
                    pair_names.append((pair[0], pair[0]))
                    data.append((img1, img2))
                    targets.append(same)
                for pair in unmatched_pairs:
                    img1, img2, same = (self._get_path(pair[0], pair[1]), self._get_path(pair[2], pair[3]), 0)
                    pair_names.append((pair[0], pair[2]))
                    data.append((img1, img2))
                    targets.append(same)
        return (pair_names, data, targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is `0` for different indentities and `1` for same identities.
        """
        img1, img2 = self.data[index]
        img1, img2 = (self._loader(img1), self._loader(img2))
        target = self.targets[index]
        if self.transform is not None:
            img1, img2 = (self.transform(img1), self.transform(img2))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img1, img2, target)