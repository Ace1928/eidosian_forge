from __future__ import annotations
import tokenize
from typing import TYPE_CHECKING
import warnings
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.parsing import tokenize_string
from pandas.core.computation.scope import ensure_scope
from pandas.core.generic import NDFrame
from pandas.io.formats.printing import pprint_thing
def _check_engine(engine: str | None) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : str
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
    from pandas.core.computation.check import NUMEXPR_INSTALLED
    from pandas.core.computation.expressions import USE_NUMEXPR
    if engine is None:
        engine = 'numexpr' if USE_NUMEXPR else 'python'
    if engine not in ENGINES:
        valid_engines = list(ENGINES.keys())
        raise KeyError(f"Invalid engine '{engine}' passed, valid engines are {valid_engines}")
    if engine == 'numexpr' and (not NUMEXPR_INSTALLED):
        raise ImportError("'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed")
    return engine