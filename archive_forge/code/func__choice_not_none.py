import cupy
def _choice_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg