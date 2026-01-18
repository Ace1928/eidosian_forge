from collections import Counter
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import build_xarray_data, flatten_chains, get_logger
from cmdstanpy.utils.stancsv import scan_generic_csv
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB
def _assemble_generated_quantities(self) -> None:
    if self._draws.shape != (0,):
        return
    _, num_draws = self._draws_start(inc_warmup=True)
    gq_sample: np.ndarray = np.empty((num_draws, self.chains, len(self.column_names)), dtype=float, order='F')
    for chain in range(self.chains):
        with open(self.runset.csv_files[chain], 'r') as fd:
            lines = (line for line in fd if not line.startswith('#'))
            gq_sample[:, chain, :] = np.loadtxt(lines, dtype=np.ndarray, ndmin=2, skiprows=1, delimiter=',')
    self._draws = gq_sample