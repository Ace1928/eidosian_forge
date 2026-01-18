from typing import List, Optional, Union
class GeoFilter(Filter):
    METERS = 'm'
    KILOMETERS = 'km'
    FEET = 'ft'
    MILES = 'mi'

    def __init__(self, field: str, lon: float, lat: float, radius: float, unit: str=KILOMETERS) -> None:
        Filter.__init__(self, 'GEOFILTER', field, lon, lat, radius, unit)