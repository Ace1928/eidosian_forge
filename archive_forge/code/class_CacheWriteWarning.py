import warnings
class CacheWriteWarning(ModelWarning):
    """
    Attempting to write to a read-only cached value
    """
    pass