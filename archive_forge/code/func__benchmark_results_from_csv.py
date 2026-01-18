import argparse
import contextlib
import copy
import csv
import functools
import glob
import itertools
import logging
import math
import os
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark
def _benchmark_results_from_csv(filename: str) -> List[Tuple[Dict[str, Any], Any]]:
    parts = os.path.basename(filename).split('.')
    env = ''
    description = ''
    if len(parts) == 3:
        env = parts[1]
        description = parts[0]
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if description != '' and row['description'] not in BASELINE_DESCRIPTIONS:
                row['description'] = description
            task_spec = benchmark.utils.common.TaskSpec(stmt='', setup='', global_setup='', label=row['label'], sub_label=row['sub_label'], description=row['description'], env=env, num_threads=int(row['num_threads']))
            measurement = benchmark.utils.common.Measurement(number_per_run=1, raw_times=[float(row['runtime_us']) / (1000.0 * 1000)], task_spec=task_spec)
            measurement.mem_use = float(row['mem_use_mb'])
            data.append(({META_ALGORITHM: row['algorithm'] if row['algorithm'] != '' else None}, measurement))
    return data