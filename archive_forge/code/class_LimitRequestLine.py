class LimitRequestLine(ParseException):

    def __init__(self, size, max_size):
        self.size = size
        self.max_size = max_size

    def __str__(self):
        return 'Request Line is too large (%s > %s)' % (self.size, self.max_size)