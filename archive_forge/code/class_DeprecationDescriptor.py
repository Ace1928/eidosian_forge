import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
class DeprecationDescriptor:
    """
    Describe deprecated parameter.

    Parameters
    ----------
    parameter : type[Parameter]
        Deprecated parameter.
    new_parameter : type[Parameter], optional
        If there's a replacement parameter for the deprecated one, specify it here.
    when_removed : str, optional
        If known, the exact release when the deprecated parameter is planned to be removed.
    """
    _parameter: type['Parameter']
    _new_parameter: Optional[type['Parameter']]
    _when_removed: str

    def __init__(self, parameter: type['Parameter'], new_parameter: Optional[type['Parameter']]=None, when_removed: Optional[str]=None):
        self._parameter = parameter
        self._new_parameter = new_parameter
        self._when_removed = 'a future' if when_removed is None else when_removed

    def deprecation_message(self, use_envvar_names: bool=False) -> str:
        """
        Generate a message to be used in a warning raised when using the deprecated parameter.

        Parameters
        ----------
        use_envvar_names : bool, default: False
            Whether to use environment variable names in the warning. If ``True``, both
            ``self._parameter`` and ``self._new_parameter`` have to be a type of ``EnvironmentVariable``.

        Returns
        -------
        str
        """
        name = cast('EnvironmentVariable', self._parameter).varname if use_envvar_names else self._parameter.__name__
        msg = f"'{name}' is deprecated and will be removed in {self._when_removed} version."
        if self._new_parameter is not None:
            new_name = cast('EnvironmentVariable', self._new_parameter).varname if use_envvar_names else self._new_parameter.__name__
            msg += f" Use '{new_name}' instead."
        return msg