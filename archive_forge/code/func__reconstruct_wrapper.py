import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
def _reconstruct_wrapper(_pickled_object, keep_wrapper):
    obj = loads(_pickled_object)
    return _wrap_non_picklable_objects(obj, keep_wrapper)