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
def _fail_if_regressions(results: List[Any], reference: List[Any], atol_s: float, rtol: float) -> None:

    def get_measurement_id(r):
        return (r[0].get(META_ALGORITHM, '').partition('@')[0], r[1].task_spec.label, r[1].task_spec.sub_label, r[1].task_spec.env)
    id_to_result = {}
    for r in results:
        id_to_result[get_measurement_id(r)] = r[1]
    num_better = 0
    num_worse = 0
    num_nochange = 0
    num_unk = 0
    reference_set = set()
    for ref in reference:
        if ref[1].task_spec.description in BASELINE_DESCRIPTIONS:
            continue
        benchmark_id = get_measurement_id(ref)
        if benchmark_id in reference_set:
            raise ValueError(f'Duplicate benchmark in reference for {benchmark_id}')
        reference_set.add(benchmark_id)
        if benchmark_id not in id_to_result:
            num_unk += 1
            continue
        res = id_to_result[benchmark_id]
        if abs(ref[1].mean - res.mean) - rtol * ref[1].mean > atol_s:
            is_now_better = res.mean < ref[1].mean
            if is_now_better:
                num_better += 1
            else:
                num_worse += 1
            cmp = 'IMPROVED' if is_now_better else 'REGRESS '
            print(cmp, benchmark_id, f'ref={ref[1].mean}', f'now={res.mean}')
        else:
            num_nochange += 1
    print('Regression test summary:')
    print(f'  Better   : {num_better}')
    print(f'  No change: {num_nochange}')
    print(f'  Worse    : {num_worse}')
    if num_unk > 0:
        print(f'  (no ref) : {num_unk}')
    benchmarks_run = num_better + num_nochange + num_worse
    if num_worse > 1:
        raise RuntimeError('At least one benchmark regressed!')
    elif num_unk == benchmarks_run:
        raise RuntimeError('No reference found')
    elif benchmarks_run == 0:
        raise RuntimeError('No benchmark was run')