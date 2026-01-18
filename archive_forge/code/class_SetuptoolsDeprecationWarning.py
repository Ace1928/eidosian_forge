import os
import warnings
from datetime import date
from inspect import cleandoc
from textwrap import indent
from typing import Optional, Tuple
class SetuptoolsDeprecationWarning(SetuptoolsWarning):
    """
    Base class for warning deprecations in ``setuptools``

    This class is not derived from ``DeprecationWarning``, and as such is
    visible by default.
    """