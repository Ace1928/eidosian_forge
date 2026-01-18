import glob
import itertools
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def _assert_frame_approx_equal(df, df2, *, atol):
    assert len(df) == len(df2)
    for (i1, row1), (i2, row2) in zip(df.sort_index().iterrows(), df2.sort_index().iterrows()):
        assert i1 == i2
        for k in set(row1.keys()) | set(row2.keys()):
            v1 = row1[k]
            v2 = row2[k]
            if isinstance(v1, (float, np.ndarray)):
                np.testing.assert_allclose(v1, v2, atol=atol)
            else:
                assert v1 == v2, k