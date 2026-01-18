from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class SettingWithCopyWarning(Warning):
    """
    Warning raised when trying to set on a copied slice from a ``DataFrame``.

    The ``mode.chained_assignment`` needs to be set to set to 'warn.'
    'Warn' is the default option. This can happen unintentionally when
    chained indexing.

    For more information on evaluation order,
    see :ref:`the user guide<indexing.evaluation_order>`.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df.loc[0:3]['A'] = 'a' # doctest: +SKIP
    ... # SettingWithCopyWarning: A value is trying to be set on a copy of a...
    """