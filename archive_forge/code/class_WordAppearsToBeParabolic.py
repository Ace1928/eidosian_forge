class WordAppearsToBeParabolic(DrillGeodesicError):

    def __init__(self, word, trace):
        self.word = word
        self.trace = trace
        super().__init__('Attempting to drill a geodesic corresponding to a matrix that could be parabolic. Word: %s, trace: %r.' % (word, trace))