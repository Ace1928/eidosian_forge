from functools import wraps
class NoOpCallback(Callback):
    """
    This implementation of Callback does exactly nothing
    """

    def call(self, *args, **kwargs):
        return None