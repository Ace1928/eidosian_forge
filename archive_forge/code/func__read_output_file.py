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
def _read_output_file(path):
    """Read Stan csv file to ndarray."""
    comments = []
    data = []
    columns = None
    with open(path, 'rb') as f_obj:
        for line in f_obj:
            if line.startswith(b'#'):
                comments.append(line.strip().decode('utf-8'))
                continue
            columns = {key: idx for idx, key in enumerate(line.strip().decode('utf-8').split(','))}
            break
        for line in f_obj:
            line = line.strip()
            if line.startswith(b'#'):
                comments.append(line.decode('utf-8'))
                continue
            if line:
                data.append(np.array(line.split(b','), dtype=np.float64))
    return (columns, np.array(data, dtype=np.float64), comments)