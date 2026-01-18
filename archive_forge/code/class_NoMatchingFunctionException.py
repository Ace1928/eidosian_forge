class NoMatchingFunctionException(FunctionResolutionError):

    def __init__(self, name):
        super(NoMatchingFunctionException, self).__init__(u'No function "{0}" matches supplied arguments'.format(name))