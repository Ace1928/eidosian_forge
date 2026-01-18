import itertools
import multiprocessing
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def _summary_stats(row):
    D = 4
    row['e_u'] = np.sum(row['pure_probs'] ** 2)
    row['u_u'] = np.sum(row['pure_probs']) / D
    row['m_u'] = np.sum(row['pure_probs'] * row['sampled_probs'])
    row['y'] = row['m_u'] - row['u_u']
    row['x'] = row['e_u'] - row['u_u']
    row['numerator'] = row['x'] * row['y']
    row['denominator'] = row['x'] ** 2
    return row