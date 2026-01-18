class GeoValue(Value):

    def __init__(self, lon, lat, radius, unit='km'):
        self.lon = lon
        self.lat = lat
        self.radius = radius
        self.unit = unit