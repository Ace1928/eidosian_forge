import functools
import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, cast, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from .utils import _read_pfm, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
class Middlebury2014Stereo(StereoMatchingDataset):
    """Publicly available scenes from the Middlebury dataset `2014 version <https://vision.middlebury.edu/stereo/data/scenes2014/>`.

    The dataset mostly follows the original format, without containing the ambient subdirectories.  : ::

        root
            Middlebury2014
                train
                    scene1-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    scene2-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    ...
                additional
                    scene1-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    ...
                test
                    scene1
                        calib.txt
                        im{0,1}.png
                    scene2
                        calib.txt
                        im{0,1}.png
                    ...

    Args:
        root (string): Root directory of the Middleburry 2014 Dataset.
        split (string, optional): The dataset split of scenes, either "train" (default), "test", or "additional"
        use_ambient_views (boolean, optional): Whether to use different expose or lightning views when possible.
            The dataset samples with equal probability between ``[im1.png, im1E.png, im1L.png]``.
        calibration (string, optional): Whether or not to use the calibrated (default) or uncalibrated scenes.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        download (boolean, optional): Whether or not to download the dataset in the ``root`` directory.
    """
    splits = {'train': ['Adirondack', 'Jadeplant', 'Motorcycle', 'Piano', 'Pipes', 'Playroom', 'Playtable', 'Recycle', 'Shelves', 'Vintage'], 'additional': ['Backpack', 'Bicycle1', 'Cable', 'Classroom1', 'Couch', 'Flowers', 'Mask', 'Shopvac', 'Sticks', 'Storage', 'Sword1', 'Sword2', 'Umbrella'], 'test': ['Plants', 'Classroom2E', 'Classroom2', 'Australia', 'DjembeL', 'CrusadeP', 'Crusade', 'Hoops', 'Bicycle2', 'Staircase', 'Newkuba', 'AustraliaP', 'Djembe', 'Livingroom', 'Computer']}
    _has_built_in_disparity_mask = True

    def __init__(self, root: str, split: str='train', calibration: Optional[str]='perfect', use_ambient_views: bool=False, transforms: Optional[Callable]=None, download: bool=False) -> None:
        super().__init__(root, transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test', 'additional'))
        self.split = split
        if calibration:
            verify_str_arg(calibration, 'calibration', valid_values=('perfect', 'imperfect', 'both', None))
            if split == 'test':
                raise ValueError("Split 'test' has only no calibration settings, please set `calibration=None`.")
        elif split != 'test':
            raise ValueError(f"Split '{split}' has calibration settings, however None was provided as an argument.\nSetting calibration to 'perfect' for split '{split}'. Available calibration settings are: 'perfect', 'imperfect', 'both'.")
        if download:
            self._download_dataset(root)
        root = Path(root) / 'Middlebury2014'
        if not os.path.exists(root / split):
            raise FileNotFoundError(f'The {split} directory was not found in the provided root directory')
        split_scenes = self.splits[split]
        if not any((scene.startswith(s) for scene in os.listdir(root / split) for s in split_scenes)):
            raise FileNotFoundError(f'Provided root folder does not contain any scenes from the {split} split.')
        calibrartion_suffixes = {None: [''], 'perfect': ['-perfect'], 'imperfect': ['-imperfect'], 'both': ['-perfect', '-imperfect']}[calibration]
        for calibration_suffix in calibrartion_suffixes:
            scene_pattern = '*' + calibration_suffix
            left_img_pattern = str(root / split / scene_pattern / 'im0.png')
            right_img_pattern = str(root / split / scene_pattern / 'im1.png')
            self._images += self._scan_pairs(left_img_pattern, right_img_pattern)
            if split == 'test':
                self._disparities = list(((None, None) for _ in self._images))
            else:
                left_dispartity_pattern = str(root / split / scene_pattern / 'disp0.pfm')
                right_dispartity_pattern = str(root / split / scene_pattern / 'disp1.pfm')
                self._disparities += self._scan_pairs(left_dispartity_pattern, right_dispartity_pattern)
        self.use_ambient_views = use_ambient_views

    def _read_img(self, file_path: Union[str, Path]) -> Image.Image:
        """
        Function that reads either the original right image or an augmented view when ``use_ambient_views`` is True.
        When ``use_ambient_views`` is True, the dataset will return at random one of ``[im1.png, im1E.png, im1L.png]``
        as the right image.
        """
        ambient_file_paths: List[Union[str, Path]]
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if file_path.name == 'im1.png' and self.use_ambient_views:
            base_path = file_path.parent
            ambient_file_paths = list((base_path / view_name for view_name in ['im1E.png', 'im1L.png']))
            ambient_file_paths = list(filter(lambda p: os.path.exists(p), ambient_file_paths))
            ambient_file_paths.append(file_path)
            file_path = random.choice(ambient_file_paths)
        return super()._read_img(file_path)

    def _read_disparity(self, file_path: str) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
        if file_path is None:
            return (None, None)
        disparity_map = _read_pfm_file(file_path)
        disparity_map = np.abs(disparity_map)
        disparity_map[disparity_map == np.inf] = 0
        valid_mask = (disparity_map > 0).squeeze(0)
        return (disparity_map, valid_mask)

    def _download_dataset(self, root: str) -> None:
        base_url = 'https://vision.middlebury.edu/stereo/data/scenes2014/zip'
        root = Path(root) / 'Middlebury2014'
        split_name = self.split
        if split_name != 'test':
            for split_scene in self.splits[split_name]:
                split_root = root / split_name
                for calibration in ['perfect', 'imperfect']:
                    scene_name = f'{split_scene}-{calibration}'
                    scene_url = f'{base_url}/{scene_name}.zip'
                    print(f'Downloading {scene_url}')
                    if not (split_root / scene_name).exists():
                        download_and_extract_archive(url=scene_url, filename=f'{scene_name}.zip', download_root=str(split_root), remove_finished=True)
        else:
            os.makedirs(root / 'test')
            if any((s not in os.listdir(root / 'test') for s in self.splits['test'])):
                test_set_url = 'https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-F.zip'
                download_and_extract_archive(url=test_set_url, download_root=str(root), remove_finished=True)
                for scene_dir, scene_names, _ in os.walk(str(root / 'MiddEval3/testF')):
                    for scene in scene_names:
                        scene_dst_dir = root / 'test'
                        scene_src_dir = Path(scene_dir) / scene
                        os.makedirs(scene_dst_dir, exist_ok=True)
                        shutil.move(str(scene_src_dir), str(scene_dst_dir))
                shutil.rmtree(str(root / 'MiddEval3'))

    def __getitem__(self, index: int) -> T2:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img_left, img_right, disparity, valid_mask)``.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images.
            ``valid_mask`` is implicitly ``None`` for `split=test`.
        """
        return cast(T2, super().__getitem__(index))