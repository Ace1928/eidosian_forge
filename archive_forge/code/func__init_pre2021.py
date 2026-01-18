import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def _init_pre2021(self) -> None:
    """Initialize based on 2017-2019 layout"""
    self.categories_index = {'super': {}}
    cat_index = 0
    super_categories = sorted(os.listdir(self.root))
    for sindex, scat in enumerate(super_categories):
        self.categories_index['super'][scat] = sindex
        subcategories = sorted(os.listdir(os.path.join(self.root, scat)))
        for subcat in subcategories:
            if self.version == '2017':
                subcat_i = cat_index
                cat_index += 1
            else:
                try:
                    subcat_i = int(subcat)
                except ValueError:
                    raise RuntimeError(f'Unexpected non-numeric dir name: {subcat}')
            if subcat_i >= len(self.categories_map):
                old_len = len(self.categories_map)
                self.categories_map.extend([{}] * (subcat_i - old_len + 1))
                self.all_categories.extend([''] * (subcat_i - old_len + 1))
            if self.categories_map[subcat_i]:
                raise RuntimeError(f'Duplicate category {subcat}')
            self.categories_map[subcat_i] = {'super': sindex}
            self.all_categories[subcat_i] = os.path.join(scat, subcat)
    for cindex, c in enumerate(self.categories_map):
        if not c:
            raise RuntimeError(f'Missing category {cindex}')