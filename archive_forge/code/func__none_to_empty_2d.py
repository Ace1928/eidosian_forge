import cupy
def _none_to_empty_2d(arg):
    if arg is None:
        return cupy.zeros((0, 0))
    else:
        return arg