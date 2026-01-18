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
def _previous_draws(self, inc_warmup: bool) -> np.ndarray:
    """
        Extract the draws from self.previous_fit.
        Return is always 3-d
        """
    p_fit = self.previous_fit
    if isinstance(p_fit, CmdStanMCMC):
        return p_fit.draws(inc_warmup=inc_warmup)
    elif isinstance(p_fit, CmdStanMLE):
        if inc_warmup and p_fit._save_iterations:
            return p_fit.optimized_iterations_np[:, None]
        return np.atleast_2d(p_fit.optimized_params_np)[:, None]
    else:
        if inc_warmup:
            return np.vstack([p_fit.variational_params_np, p_fit.variational_sample])[:, None]
        return p_fit.variational_sample[:, None]