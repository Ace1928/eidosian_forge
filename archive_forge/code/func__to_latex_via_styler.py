from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def _to_latex_via_styler(self, buf=None, *, hide: dict | list[dict] | None=None, relabel_index: dict | list[dict] | None=None, format: dict | list[dict] | None=None, format_index: dict | list[dict] | None=None, render_kwargs: dict | None=None):
    """
        Render object to a LaTeX tabular, longtable, or nested table.

        Uses the ``Styler`` implementation with the following, ordered, method chaining:

        .. code-block:: python
           styler = Styler(DataFrame)
           styler.hide(**hide)
           styler.relabel_index(**relabel_index)
           styler.format(**format)
           styler.format_index(**format_index)
           styler.to_latex(buf=buf, **render_kwargs)

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        hide : dict, list of dict
            Keyword args to pass to the method call of ``Styler.hide``. If a list will
            call the method numerous times.
        relabel_index : dict, list of dict
            Keyword args to pass to the method of ``Styler.relabel_index``. If a list
            will call the method numerous times.
        format : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format``. If a list will
            call the method numerous times.
        format_index : dict, list of dict
            Keyword args to pass to the method call of ``Styler.format_index``. If a
            list will call the method numerous times.
        render_kwargs : dict
            Keyword args to pass to the method call of ``Styler.to_latex``.

        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns None.
        """
    from pandas.io.formats.style import Styler
    self = cast('DataFrame', self)
    styler = Styler(self, uuid='')
    for kw_name in ['hide', 'relabel_index', 'format', 'format_index']:
        kw = vars()[kw_name]
        if isinstance(kw, dict):
            getattr(styler, kw_name)(**kw)
        elif isinstance(kw, list):
            for sub_kw in kw:
                getattr(styler, kw_name)(**sub_kw)
    render_kwargs = {} if render_kwargs is None else render_kwargs
    if render_kwargs.pop('bold_rows'):
        styler.map_index(lambda v: 'textbf:--rwrap;')
    return styler.to_latex(buf=buf, **render_kwargs)