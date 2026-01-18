from ._multiprocessing_helpers import mp
def _my_wrap_non_picklable_objects(obj, keep_wrapper=True):
    return obj