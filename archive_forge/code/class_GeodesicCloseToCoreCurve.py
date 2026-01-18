class GeodesicCloseToCoreCurve(DrillGeodesicError):

    def __init__(self):
        super().__init__('The given geodesic is very close to a core curve and might intersect it.')