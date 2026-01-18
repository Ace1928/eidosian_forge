import csv
import os
import time
import urllib
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
from .folder import find_classes, make_dataset
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .video_utils import VideoClips
from .vision import VisionDataset
def _download_videos(self) -> None:
    """download tarballs containing the video to "tars" folder and extract them into the _split_ folder where
        split is one of the official dataset splits.

        Raises:
            RuntimeError: if download folder exists, break to prevent downloading entire dataset again.
        """
    if path.exists(self.split_folder):
        raise RuntimeError(f'The directory {self.split_folder} already exists. If you want to re-download or re-extract the images, delete the directory.')
    tar_path = path.join(self.root, 'tars')
    file_list_path = path.join(self.root, 'files')
    split_url = self._TAR_URLS[self.num_classes].format(split=self.split)
    split_url_filepath = path.join(file_list_path, path.basename(split_url))
    if not check_integrity(split_url_filepath):
        download_url(split_url, file_list_path)
    with open(split_url_filepath) as file:
        list_video_urls = [urllib.parse.quote(line, safe='/,:') for line in file.read().splitlines()]
    if self.num_download_workers == 1:
        for line in list_video_urls:
            download_and_extract_archive(line, tar_path, self.split_folder)
    else:
        part = partial(_dl_wrap, tar_path, self.split_folder)
        poolproc = Pool(self.num_download_workers)
        poolproc.map(part, list_video_urls)