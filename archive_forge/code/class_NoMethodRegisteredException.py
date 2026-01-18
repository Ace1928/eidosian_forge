class NoMethodRegisteredException(MethodResolutionError):

    def __init__(self, name, receiver):
        super(NoMethodRegisteredException, self).__init__(u'Unknown method "{0}" for receiver {1}'.format(name, receiver))