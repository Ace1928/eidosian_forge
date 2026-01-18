import sys
@wraps(callback)
def adapted(*args, **kwargs):
    """Wrapper for third party callbacks that discards excess arguments"""
    args = args[:n_positional]
    for name in unmatched_kw:
        kwargs.pop(name)
    return callback(*args, **kwargs)