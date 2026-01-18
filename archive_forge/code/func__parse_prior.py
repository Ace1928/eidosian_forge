import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
@requires('prior_')
def _parse_prior(self):
    """Read csv paths to list of ndarrays."""
    paths = self.prior_
    if isinstance(paths, str):
        paths = [paths]
    chain_data = []
    columns = None
    for path in paths:
        output_data = _read_output(path)
        chain_data.append(output_data)
        if columns is None:
            columns = output_data
    self.prior = ([item['sample'] for item in chain_data], [item['sample_warmup'] for item in chain_data])
    self.prior_columns = columns['sample_columns']
    self.sample_stats_prior_columns = columns['sample_stats_columns']
    attrs = {}
    for item in chain_data:
        for key, value in item['configuration_info'].items():
            if key not in attrs:
                attrs[key] = []
            attrs[key].append(value)
    self.attrs_prior = attrs