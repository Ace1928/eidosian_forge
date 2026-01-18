import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
@dataclass(frozen=True)
class TransformerFromCRS(TransformerMaker):
    """
    .. versionadded:: 3.1.0

    .. versionadded:: 3.4.0 force_over

    Generates a Cython _Transformer class from input CRS data.
    """
    crs_from: bytes
    crs_to: bytes
    always_xy: bool
    area_of_interest: Optional[AreaOfInterest]
    authority: Optional[str]
    accuracy: Optional[str]
    allow_ballpark: Optional[bool]
    force_over: bool = False
    only_best: Optional[bool] = None

    def __call__(self) -> _Transformer:
        """
        Returns
        -------
        _Transformer
        """
        return _Transformer.from_crs(self.crs_from, self.crs_to, always_xy=self.always_xy, area_of_interest=self.area_of_interest, authority=self.authority, accuracy=self.accuracy, allow_ballpark=self.allow_ballpark, force_over=self.force_over, only_best=self.only_best)