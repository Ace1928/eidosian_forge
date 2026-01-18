class SkeletonPoint(object):
    location = None
    shapeMoments = None
    shapeDirs = None
    molFeatures = None
    featmapFeatures = None
    fracVol = 0.0

    def __init__(self, *args, **kwargs):
        self._initMemberData()
        self.location = kwargs.get('location', None)

    def _initMemberData(self):
        self.shapeMoments = (0.0,) * 3
        self.shapeDirs = [None] * 3
        self.molFeatures = []
        self.featmapFeatures = []