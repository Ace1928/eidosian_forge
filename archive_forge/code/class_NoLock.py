import six
class NoLock(object):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass