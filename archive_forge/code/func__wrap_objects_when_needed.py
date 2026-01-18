import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
def _wrap_objects_when_needed(obj):
    need_wrap = '__main__' in getattr(obj, '__module__', '')
    if isinstance(obj, partial):
        return partial(_wrap_objects_when_needed(obj.func), *[_wrap_objects_when_needed(a) for a in obj.args], **{k: _wrap_objects_when_needed(v) for k, v in obj.keywords.items()})
    if callable(obj):
        func_code = getattr(obj, '__code__', '')
        need_wrap |= getattr(func_code, 'co_flags', 0) & inspect.CO_NESTED
        func_name = getattr(obj, '__name__', '')
        need_wrap |= '<lambda>' in func_name
    if not need_wrap:
        return obj
    wrapped_obj = WRAP_CACHE.get(obj)
    if wrapped_obj is None:
        wrapped_obj = _wrap_non_picklable_objects(obj, keep_wrapper=False)
        WRAP_CACHE[obj] = wrapped_obj
    return wrapped_obj