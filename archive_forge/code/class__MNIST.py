import logging
import os
import random
import time
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
class _MNIST(Dataset):
    """Carbon copy of ``tests_pytorch.helpers.datasets.MNIST``.

    We cannot import the tests as they are not distributed with the package.
    See https://github.com/Lightning-AI/lightning/pull/7614#discussion_r671183652 for more context.

    .. warning::  This is meant for testing/debugging and is experimental.

    """
    RESOURCES = ('https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt', 'https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt')
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    cache_folder_name = 'complete'

    def __init__(self, root: str, train: bool=True, normalize: tuple=(0.1307, 0.3081), download: bool=True, **kwargs: Any) -> None:
        super().__init__()
        self.root = root
        self.train = train
        self.normalize = normalize
        self.prepare_data(download)
        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = self._try_load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])
        if self.normalize is not None and len(self.normalize) == 2:
            img = self.normalize_tensor(img, *self.normalize)
        return (img, target)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root, 'MNIST', self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool=True) -> None:
        if download and (not self._check_exists(self.cached_folder_path)):
            self._download(self.cached_folder_path)
        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError('Dataset not found.')

    def _download(self, data_folder: str) -> None:
        os.makedirs(data_folder, exist_ok=True)
        for url in self.RESOURCES:
            logging.info(f'Downloading {url}')
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)

    @staticmethod
    def _try_load(path_data: str, trials: int=30, delta: float=1.0) -> Tuple[Tensor, Tensor]:
        """Resolving loading from the same time from multiple concurrent processes."""
        res, exception = (None, None)
        assert trials, 'at least some trial has to be set'
        assert os.path.isfile(path_data), f'missing file: {path_data}'
        for _ in range(trials):
            try:
                res = torch.load(path_data)
            except Exception as ex:
                exception = ex
                time.sleep(delta * random.random())
            else:
                break
        assert res is not None
        if exception is not None:
            raise exception
        return res

    @staticmethod
    def normalize_tensor(tensor: Tensor, mean: Union[int, float]=0.0, std: Union[int, float]=1.0) -> Tensor:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)