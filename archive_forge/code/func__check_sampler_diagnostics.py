import math
import os
from io import StringIO
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP, _TMPDIR
from cmdstanpy.cmdstan_args import Method, SamplerArgs
from cmdstanpy.utils import (
from .metadata import InferenceMetadata
from .runset import RunSet
def _check_sampler_diagnostics(self) -> None:
    """
        Warn if any iterations ended in divergences or hit maxtreedepth.
        """
    if np.any(self._divergences) or np.any(self._max_treedepths):
        diagnostics = ['Some chains may have failed to converge.']
        ct_iters = self._metadata.cmdstan_config['num_samples']
        for i in range(self.runset._chains):
            if self._divergences[i] > 0:
                diagnostics.append(f'Chain {i + 1} had {self._divergences[i]} divergent transitions ({self._divergences[i] / ct_iters * 100:.1f}%)')
            if self._max_treedepths[i] > 0:
                diagnostics.append(f'Chain {i + 1} had {self._max_treedepths[i]} iterations at max treedepth ({self._max_treedepths[i] / ct_iters * 100:.1f}%)')
        diagnostics.append('Use the "diagnose()" method on the CmdStanMCMC object to see further information.')
        get_logger().warning('\n\t'.join(diagnostics))