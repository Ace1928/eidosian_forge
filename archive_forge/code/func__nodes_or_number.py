import bz2
import collections
import gzip
import inspect
import itertools
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature
from os.path import splitext
from pathlib import Path
import networkx as nx
from networkx.utils import create_py_random_state, create_random_state
def _nodes_or_number(n):
    try:
        nodes = list(range(n))
    except TypeError:
        nodes = tuple(n)
    else:
        if n < 0:
            raise nx.NetworkXError(f'Negative number of nodes not valid: {n}')
    return (n, nodes)