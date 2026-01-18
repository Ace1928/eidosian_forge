import cupy
def _convert_to_type(X, out_type):
    return cupy.ascontiguousarray(X, dtype=out_type)