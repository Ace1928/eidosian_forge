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
def _test_function(arg1):
    import os
    file1 = os.path.join(os.getcwd(), 'file1.txt')
    file2 = os.path.join(os.getcwd(), 'file2.txt')
    file3 = os.path.join(os.getcwd(), 'file3.txt')
    file4 = os.path.join(os.getcwd(), 'subdir', 'file4.txt')
    os.mkdir('subdir')
    for filename in [file1, file2, file3, file4]:
        with open(filename, 'wt') as fp:
            fp.write('%d' % arg1)
    return (file1, file2, os.path.join(os.getcwd(), 'subdir'))