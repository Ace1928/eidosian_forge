import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
@dataclass
class ClickFold:
    """A structure containing information about generated user-click data."""
    X: sparse.csr_matrix
    y: npt.NDArray[np.int32]
    qid: npt.NDArray[np.int32]
    score: npt.NDArray[np.float32]
    click: npt.NDArray[np.int32]
    pos: npt.NDArray[np.int64]