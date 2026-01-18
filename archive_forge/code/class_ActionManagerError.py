import enum
class ActionManagerError(Exception):
    """
    An exception used when an error occurs within an ActionManager.
    """

    def __init__(self, *args, **kargs):
        Exception.__init__(self, *args, **kargs)