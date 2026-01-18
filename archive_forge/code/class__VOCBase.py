import collections
import os
from xml.etree.ElementTree import Element as ET_Element
from .vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(self, root: str, year: str='2012', image_set: str='train', download: bool=False, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None, transforms: Optional[Callable]=None):
        super().__init__(root, transforms, transform, target_transform)
        self.year = verify_str_arg(year, 'year', valid_values=[str(yr) for yr in range(2007, 2013)])
        valid_image_sets = ['train', 'trainval', 'val']
        if year == '2007':
            valid_image_sets.append('test')
        self.image_set = verify_str_arg(image_set, 'image_set', valid_image_sets)
        key = '2007-test' if year == '2007' and image_set == 'test' else year
        dataset_year_dict = DATASET_YEAR_DICT[key]
        self.url = dataset_year_dict['url']
        self.filename = dataset_year_dict['filename']
        self.md5 = dataset_year_dict['md5']
        base_dir = dataset_year_dict['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
        splits_dir = os.path.join(voc_root, 'ImageSets', self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
        image_dir = os.path.join(voc_root, 'JPEGImages')
        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)