from typing import List
from redis import DataError
class GeoShapeField(Field):
    """
    GeoShapeField is used to enable within/contain indexing/searching
    """
    SPHERICAL = 'SPHERICAL'
    FLAT = 'FLAT'

    def __init__(self, name: str, coord_system=None, **kwargs):
        args = [Field.GEOSHAPE]
        if coord_system:
            args.append(coord_system)
        Field.__init__(self, name, args=args, **kwargs)