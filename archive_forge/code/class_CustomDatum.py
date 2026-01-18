from typing import Any, Optional, Union
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
class CustomDatum(Datum):
    """
    .. versionadded:: 2.5.0

    Class to build a datum based on an ellipsoid and prime meridian.
    """

    def __new__(cls, name: str='undefined', ellipsoid: Any='WGS 84', prime_meridian: Any='Greenwich'):
        """
        Parameters
        ----------
        name: str, default="undefined"
            Name of the datum.
        ellipsoid: Any, default="WGS 84"
            Anything accepted by :meth:`pyproj.crs.Ellipsoid.from_user_input`
            or a :class:`pyproj.crs.datum.CustomEllipsoid`.
        prime_meridian: Any, default="Greenwich"
            Anything accepted by :meth:`pyproj.crs.PrimeMeridian.from_user_input`.
        """
        datum_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'GeodeticReferenceFrame', 'name': name, 'ellipsoid': Ellipsoid.from_user_input(ellipsoid).to_json_dict(), 'prime_meridian': PrimeMeridian.from_user_input(prime_meridian).to_json_dict()}
        return cls.from_json_dict(datum_json)