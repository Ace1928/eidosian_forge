import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union
import PIL
import torch
from .utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from .vision import VisionDataset
def _load_csv(self, filename: str, header: Optional[int]=None) -> CSV:
    with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))
    if header is not None:
        headers = data[header]
        data = data[header + 1:]
    else:
        headers = []
    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]
    return CSV(headers, indices, torch.tensor(data_int))