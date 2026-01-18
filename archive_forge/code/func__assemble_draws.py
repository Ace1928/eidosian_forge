from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils.data_munging import build_xarray_data
from cmdstanpy.utils.stancsv import scan_generic_csv
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
def _assemble_draws(self) -> None:
    if self._draws.shape != (0,):
        return
    with open(self._runset.csv_files[0], 'r') as fd:
        while fd.readline().startswith('#'):
            pass
        self._draws = np.loadtxt(fd, dtype=float, ndmin=2, delimiter=',', comments='#')