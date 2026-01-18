class mock_rados(object):

    class ioctx(object):

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            return False

        def close(self, *args, **kwargs):
            pass

    class Rados(object):

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self, *args, **kwargs):
            return self

        def __exit__(self, *args, **kwargs):
            return False

        def connect(self, *args, **kwargs):
            pass

        def open_ioctx(self, *args, **kwargs):
            return mock_rados.ioctx()

        def shutdown(self, *args, **kwargs):
            pass