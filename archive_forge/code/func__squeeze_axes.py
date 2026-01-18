from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def _squeeze_axes(shape: Sequence[int], axes: str | Sequence[str], /, skip: str | Sequence[str] | None=None) -> tuple[tuple[int, ...], str | Sequence[str], tuple[bool, ...]]:
    """Return shape and axes with length-1 dimensions removed.

    Remove unused dimensions unless their axes are listed in `skip`.

    Parameters:
        shape:
            Sequence of dimension sizes.
        axes:
            Character codes for dimensions in `shape`.
        skip:
            Character codes for dimensions whose length-1 dimensions are
            not removed. The default is 'XY'.

    Returns:
        shape:
            Sequence of dimension sizes with length-1 dimensions removed.
        axes:
            Character codes for dimensions in output `shape`.
        squeezed:
            Dimensions were kept (True) or removed (False).

    Examples:
        >>> _squeeze_axes((5, 1, 2, 1, 1), 'TZYXC')
        ((5, 2, 1), 'TYX', (True, False, True, True, False))
        >>> _squeeze_axes((1,), 'Q')
        ((1,), 'Q', (True,))

    """
    if len(shape) != len(axes):
        raise ValueError('dimensions of axes and shape do not match')
    if not axes:
        return (tuple(shape), axes, ())
    if skip is None:
        skip = ('X', 'Y', 'width', 'height', 'length')
    squeezed: list[bool] = []
    shape_squeezed: list[int] = []
    axes_squeezed: list[str] = []
    for size, ax in zip(shape, axes):
        if size > 1 or ax in skip:
            squeezed.append(True)
            shape_squeezed.append(size)
            axes_squeezed.append(ax)
        else:
            squeezed.append(False)
    if len(shape_squeezed) == 0:
        squeezed[-1] = True
        shape_squeezed.append(shape[-1])
        axes_squeezed.append(axes[-1])
    if isinstance(axes, str):
        axes = ''.join(axes_squeezed)
    else:
        axes = tuple(axes_squeezed)
    return (tuple(shape_squeezed), axes, tuple(squeezed))