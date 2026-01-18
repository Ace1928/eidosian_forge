import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
class CustomConstructorCRS(CRS):
    """
    This class is a base class for CRS classes
    that use a different constructor than the main CRS class.

    .. versionadded:: 3.2.0

    See: https://github.com/pyproj4/pyproj/issues/847
    """

    @property
    def _expected_types(self) -> tuple[str, ...]:
        """
        These are the type names of the CRS class
        that are expected when using the from_* methods.
        """
        raise NotImplementedError

    def _check_type(self):
        """
        This validates that the type of the CRS is expected
        when using the from_* methods.
        """
        if self.type_name not in self._expected_types:
            raise CRSError(f'Invalid type {self.type_name}. Expected {self._expected_types}.')

    @classmethod
    def from_user_input(cls, value: Any, **kwargs) -> 'CRS':
        """
        Initialize a CRS class instance with:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class

        Parameters
        ----------
        value : obj
            A Python int, dict, or str.

        Returns
        -------
        CRS
        """
        if isinstance(value, cls):
            return value
        crs = cls.__new__(cls)
        super(CustomConstructorCRS, crs).__init__(value, **kwargs)
        crs._check_type()
        return crs

    @property
    def geodetic_crs(self) -> Optional['CRS']:
        """
        .. versionadded:: 2.2.0

        Returns
        -------
        CRS:
            The geodeticCRS / geographicCRS from the CRS.

        """
        return None if self._crs.geodetic_crs is None else CRS(self._crs.geodetic_crs)

    @property
    def source_crs(self) -> Optional['CRS']:
        """
        The base CRS of a BoundCRS or a DerivedCRS/ProjectedCRS,
        or the source CRS of a CoordinateOperation.

        Returns
        -------
        CRS
        """
        return None if self._crs.source_crs is None else CRS(self._crs.source_crs)

    @property
    def target_crs(self) -> Optional['CRS']:
        """
        .. versionadded:: 2.2.0

        Returns
        -------
        CRS:
            The hub CRS of a BoundCRS or the target CRS of a CoordinateOperation.

        """
        return None if self._crs.target_crs is None else CRS(self._crs.target_crs)

    @property
    def sub_crs_list(self) -> list['CRS']:
        """
        If the CRS is a compound CRS, it will return a list of sub CRS objects.

        Returns
        -------
        list[CRS]
        """
        return [CRS(sub_crs) for sub_crs in self._crs.sub_crs_list]

    def to_3d(self, name: Optional[str]=None) -> 'CRS':
        """
        .. versionadded:: 3.1.0

        Convert the current CRS to the 3D version if it makes sense.

        New vertical axis attributes:
          - ellipsoidal height
          - oriented upwards
          - metre units

        Parameters
        ----------
        name: str, optional
            CRS name. Defaults to use the name of the original CRS.

        Returns
        -------
        CRS
        """
        return CRS(self._crs.to_3d(name=name))