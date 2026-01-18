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
def _not_implemented_for(g):
    if (mval is None or mval == g.is_multigraph()) and (dval is None or dval == g.is_directed()):
        raise nx.NetworkXNotImplemented(errmsg)
    return g