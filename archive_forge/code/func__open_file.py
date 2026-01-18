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
def _open_file(path):
    if isinstance(path, str):
        ext = splitext(path)[1]
    elif isinstance(path, Path):
        ext = path.suffix
        path = str(path)
    else:
        return (path, lambda: None)
    fobj = _dispatch_dict[ext](path, mode=mode)
    return (fobj, lambda: fobj.close())