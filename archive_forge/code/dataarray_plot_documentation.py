from __future__ import annotations
import functools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, overload
import numpy as np
import pandas as pd
from xarray.core.alignment import broadcast
from xarray.core.concat import concat
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
Center the pixels on the coordinates.