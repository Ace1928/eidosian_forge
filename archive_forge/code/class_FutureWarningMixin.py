from __future__ import annotations
import typing as ty
import warnings
from .deprecator import Deprecator
from .pkg_info import cmp_pkg_version
class FutureWarningMixin:
    """Insert FutureWarning for object creation

    Examples
    --------
    >>> class C: pass
    >>> class D(FutureWarningMixin, C):
    ...     warn_message = "Please, don't use this class"

    Record the warning

    >>> with warnings.catch_warnings(record=True) as warns:
    ...     d = D()
    ...     warns[0].message.args[0]
    "Please, don't use this class"
    """
    warn_message = 'This class will be removed in future versions'

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        warnings.warn(self.warn_message, FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)