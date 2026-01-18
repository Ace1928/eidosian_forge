from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import scan_variational_csv
from cmdstanpy.utils.logging import get_logger
from .metadata import InferenceMetadata
from .runset import RunSet
def _set_variational_attrs(self, sample_csv_0: str) -> None:
    meta = scan_variational_csv(sample_csv_0)
    self._metadata = InferenceMetadata(meta)
    self._column_names: Tuple[str, ...] = meta['column_names']
    self._eta: float = meta['eta']
    self._variational_mean: np.ndarray = meta['variational_mean']
    self._variational_sample: np.ndarray = meta['variational_sample']