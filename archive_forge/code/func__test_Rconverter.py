import pytest
import textwrap
import types
import warnings
from itertools import product
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib._rinterface_capi
import rpy2.robjects
import rpy2.robjects.conversion
from .. import utils
from io import StringIO
from rpy2 import rinterface
from rpy2.robjects import r, vectors, globalenv
import rpy2.robjects.packages as rpacks
def _test_Rconverter(ipython_with_magic, clean_globalenv, dataf_py, cls):
    ipython_with_magic.user_ns['dataf_py'] = dataf_py
    ipython_with_magic.run_line_magic('Rpush', 'dataf_py')
    fromr_dataf_py = ipython_with_magic.run_line_magic('Rget', 'dataf_py')
    fromr_dataf_py_again = ipython_with_magic.run_line_magic('Rget', 'dataf_py')
    assert isinstance(fromr_dataf_py, cls)
    assert len(dataf_py) == len(fromr_dataf_py)
    ipython_with_magic.run_cell_magic('R', '-o dataf_py', 'dataf_py')
    dataf_py_roundtrip = ipython_with_magic.user_ns['dataf_py']
    assert tuple(fromr_dataf_py['x']) == tuple(dataf_py_roundtrip['x'])
    assert tuple(fromr_dataf_py['y']) == tuple(dataf_py_roundtrip['y'])