class SubshapeShape(object):
    shapes = None
    featMap = None
    keyFeat = None

    def __init__(self, *args, **kwargs):
        self._initMemberData()

    def _initMemberData(self):
        self.shapes = []