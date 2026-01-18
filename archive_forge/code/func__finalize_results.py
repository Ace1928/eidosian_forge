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
def _finalize_results(results: List[Tuple[Dict[str, Any], Any]]) -> List[Any]:
    """
    Returns a `benchmark.Compare` object, except that if we have runs
    with different algorithms, we also add the algorithm name
    in the column titles
    """
    all_algorithms: Set[str] = set()
    all_description: Set[str] = set()
    for metadata, r in results:
        algo = metadata.get(META_ALGORITHM, None)
        if algo is not None:
            all_algorithms.add(algo)
        all_description.add(r.task_spec.description)
    display_algo = len(all_algorithms) > 1
    display_descr = len(all_description) > 1
    display_results = []
    for metadata, r in results:
        algo = metadata.get(META_ALGORITHM, None)
        if algo is None:
            display_results.append(r)
        else:
            r = copy.copy(r)
            description = ''
            if display_descr:
                description = r.task_spec.description
            if display_algo:
                if display_descr:
                    description += '['
                description += algo
                if display_descr:
                    description += ']'
            r.task_spec = replace(r.task_spec, description=description)
            display_results.append(r)
    return display_results