from __future__ import annotations
from pickle import PickleBuffer
from pandas.compat._constants import PY310

    Return some 1-D `uint8` typed buffer.

    Coerces anything that does not match that description to one that does
    without copying if possible (otherwise will copy).
    