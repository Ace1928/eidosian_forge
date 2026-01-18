class ExcludedVolume(object):

    def __init__(self, featInfo, index=-1, exclusionDist=3.0):
        """
    featInfo should be a sequence of ([indices],min,max) tuples

    """
        self.index = index
        try:
            _ = len(featInfo)
        except TypeError:
            raise ValueError('featInfo argument must be a sequence of sequences')
        if not len(featInfo):
            raise ValueError('featInfo argument must non-empty')
        try:
            _, _, _ = featInfo[0]
        except (TypeError, ValueError):
            raise ValueError('featInfo elements must be 3-sequences')
        self.featInfo = featInfo[:]
        self.exclusionDist = exclusionDist
        self.pos = None