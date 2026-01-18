class NoSuchClassError(Exception):

    def __str__(self):
        return 'Unknown C++ class: %s' % self.args[0]