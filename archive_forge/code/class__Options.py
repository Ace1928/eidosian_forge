from __future__ import annotations
from contextlib import contextmanager
from typing import (
from pandas.plotting._core import _get_plot_backend
class _Options(dict):
    """
    Stores pandas plotting options.

    Allows for parameter aliasing so you can just use parameter names that are
    the same as the plot function parameters, but is stored in a canonical
    format that makes it easy to breakdown into groups later.

    Examples
    --------

    .. plot::
            :context: close-figs

             >>> np.random.seed(42)
             >>> df = pd.DataFrame({'A': np.random.randn(10),
             ...                   'B': np.random.randn(10)},
             ...                   index=pd.date_range("1/1/2000",
             ...                   freq='4MS', periods=10))
             >>> with pd.plotting.plot_params.use("x_compat", True):
             ...     _ = df["A"].plot(color="r")
             ...     _ = df["B"].plot(color="g")
    """
    _ALIASES = {'x_compat': 'xaxis.compat'}
    _DEFAULT_KEYS = ['xaxis.compat']

    def __init__(self, deprecated: bool=False) -> None:
        self._deprecated = deprecated
        super().__setitem__('xaxis.compat', False)

    def __getitem__(self, key):
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f'{key} is not a valid pandas plotting option')
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f'Cannot remove default parameter {key}')
        super().__delitem__(key)

    def __contains__(self, key) -> bool:
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self) -> None:
        """
        Reset the option store to its initial state

        Returns
        -------
        None
        """
        self.__init__()

    def _get_canonical_key(self, key):
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key, value) -> Generator[_Options, None, None]:
        """
        Temporarily set a parameter value using the with statement.
        Aliasing allowed.
        """
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value