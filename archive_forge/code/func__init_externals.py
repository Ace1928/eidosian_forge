import sys
import os
from gitdb.base import *
from gitdb.db import *
from gitdb.stream import *
def _init_externals():
    """Initialize external projects by putting them into the path"""
    if 'PYOXIDIZER' not in os.environ:
        where = os.path.join(os.path.dirname(__file__), 'ext', 'smmap')
        if os.path.exists(where):
            sys.path.append(where)
    import smmap
    del smmap