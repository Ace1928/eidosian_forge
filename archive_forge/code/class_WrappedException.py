class WrappedException(YaqlException):

    def __init__(self, exception):
        self.wrapped = exception
        super(WrappedException, self).__init__(str(exception))