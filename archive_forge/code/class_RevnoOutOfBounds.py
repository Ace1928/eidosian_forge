class RevnoOutOfBounds(InternalBzrError):
    _fmt = 'The requested revision number %(revno)d is outside of the expected boundaries (%(minimum)d <= %(maximum)d).'

    def __init__(self, revno, bounds):
        InternalBzrError.__init__(self, revno=revno, minimum=bounds[0], maximum=bounds[1])