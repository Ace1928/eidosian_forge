import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
class _VideoTimestampsDataset:
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.

    Used in VideoClips and defined at top level, so it can be
    pickled when forking.
    """

    def __init__(self, video_paths: List[str]) -> None:
        self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[float]]:
        return read_video_timestamps(self.video_paths[idx])