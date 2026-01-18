import numpy as np
from . import (
A module with platform-specific extended precision
`numpy.number` subclasses.

The subclasses are defined here (instead of ``__init__.pyi``) such
that they can be imported conditionally via the numpy's mypy plugin.
