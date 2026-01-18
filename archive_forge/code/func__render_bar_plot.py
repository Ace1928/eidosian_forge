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
def _render_bar_plot(results: List[Any], store_results_folder: str) -> None:
    if not results:
        return
    runtime: Dict[str, Dict[str, float]] = defaultdict(dict)
    memory_usage: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_descriptions: List[str] = []
    for r in results:
        if r.task_spec.description not in all_descriptions:
            if r.task_spec.description in BASELINE_DESCRIPTIONS:
                all_descriptions.insert(0, r.task_spec.description)
            else:
                all_descriptions.append(r.task_spec.description)
        runtime[r.task_spec.sub_label][r.task_spec.description] = r.mean
        memory_usage[r.task_spec.sub_label][r.task_spec.description] = r.mem_use
    all_data_mem: List[Any] = []
    all_data_run: List[Any] = []
    for key, runtime_values in runtime.items():
        memory_values = memory_usage[key]
        denom = memory_values.get(all_descriptions[0], math.inf)
        if denom == 0:
            all_data_mem.append([key] + [0] * len(all_descriptions))
        else:
            all_data_mem.append([key] + [memory_values.get(d, 0) / denom for d in all_descriptions])
        all_data_run.append([key] + [runtime_values.get(all_descriptions[0], 0) / runtime_values.get(d, math.inf) for d in all_descriptions])
    if all_descriptions[0] == '':
        all_descriptions[0] = 'baseline'
    else:
        all_descriptions[0] = f'{all_descriptions[0]} (baseline)'
    for data, filename, title in [(all_data_mem, 'mem.png', 'Memory usage (vs baseline, lower is better)'), (all_data_run, 'runtime.png', 'Runtime speedup (vs baseline, higher is better)')]:
        df = pd.DataFrame(data, columns=['Configuration'] + all_descriptions)
        df.plot(x='Configuration', kind='bar', stacked=False, title=title)
        plt.tight_layout()
        filename_full = os.path.join(store_results_folder, filename)
        plt.savefig(filename_full)
        print(f'Saved plot: {filename_full}')