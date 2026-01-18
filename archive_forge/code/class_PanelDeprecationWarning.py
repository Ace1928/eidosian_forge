from __future__ import annotations
import inspect
import os
import warnings
import param
from packaging.version import Version
class PanelDeprecationWarning(DeprecationWarning):
    """A Panel-specific ``DeprecationWarning`` subclass.
    Used to selectively filter Panel deprecations for unconditional display.
    """