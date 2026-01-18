from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def _scrape_options_class_doc(options_class):
    if not options_class.__doc__:
        return None
    doc = docscrape.NumpyDocString(options_class.__doc__)
    return _OptionsClassDoc(doc['Parameters'])