from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def _update_ctx(self, attrs: DataFrame) -> None:
    """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
    if not self.index.is_unique or not self.columns.is_unique:
        raise KeyError('`Styler.apply` and `.map` are not compatible with non-unique index or columns.')
    for cn in attrs.columns:
        j = self.columns.get_loc(cn)
        ser = attrs[cn]
        for rn, c in ser.items():
            if not c or pd.isna(c):
                continue
            css_list = maybe_convert_css_to_tuples(c)
            i = self.index.get_loc(rn)
            self.ctx[i, j].extend(css_list)