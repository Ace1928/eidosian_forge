import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
def _unpack_fit_pre_v_1_2_0(fit, items, save_warmup, dtypes):
    num_warmup = 0
    if save_warmup:
        if not fit._save_warmup:
            save_warmup = False
        else:
            num_warmup = fit.num_draws_warmup
    nchains = fit.chains
    sample = {}
    sample_warmup = {}
    stan_vars_cols = list(fit.metadata.stan_vars_cols)
    sampler_vars = fit.method_variables()
    for item in items:
        if item in stan_vars_cols:
            raw_draws = fit.stan_variable(item, inc_warmup=save_warmup)
            raw_draws = np.swapaxes(raw_draws.reshape((-1, nchains, *raw_draws.shape[1:]), order='F'), 0, 1)
        elif item in sampler_vars:
            raw_draws = np.swapaxes(sampler_vars[item], 0, 1)
        else:
            raise ValueError(f'fit data, unknown variable: {item}')
        raw_draws = raw_draws.astype(dtypes.get(item))
        if save_warmup:
            sample_warmup[item] = raw_draws[:, :num_warmup, ...]
            sample[item] = raw_draws[:, num_warmup:, ...]
        else:
            sample[item] = raw_draws
    return (sample, sample_warmup)