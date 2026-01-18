class AmbiguousMethodException(MethodResolutionError):

    def __init__(self, name, receiver):
        super(AmbiguousMethodException, self).__init__(u'Ambiguous method "{0}" for receiver {1}'.format(name, receiver))