from glob import glob
import os
from shutil import rmtree
from itertools import product
import pytest
import networkx as nx
from .... import config
from ....interfaces import utility as niu
from ... import engine as pe
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
def _test_function2(in_file, arg):
    import os
    with open(in_file, 'rt') as fp:
        in_arg = fp.read()
    file1 = os.path.join(os.getcwd(), 'file1.txt')
    file2 = os.path.join(os.getcwd(), 'file2.txt')
    file3 = os.path.join(os.getcwd(), 'file3.txt')
    files = [file1, file2, file3]
    for filename in files:
        with open(filename, 'wt') as fp:
            fp.write('%d' % arg + in_arg)
    return (file1, file2, 1)