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
class TransformerMaker(ABC):
    """
    .. versionadded:: 3.1.0

    Base class for generating new instances
    of the Cython _Transformer class for
    thread safety in the Transformer class.
    """

    @abstractmethod
    def __call__(self) -> _Transformer:
        """
        Returns
        -------
        _Transformer
        """
        raise NotImplementedError