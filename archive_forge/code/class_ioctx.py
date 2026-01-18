class ioctx(object):

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    def close(self, *args, **kwargs):
        pass