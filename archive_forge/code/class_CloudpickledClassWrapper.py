import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
class CloudpickledClassWrapper(CloudpickledObjectWrapper):

    def __init__(self, *args, **kwargs):
        self._obj = obj(*args, **kwargs)
        self._keep_wrapper = keep_wrapper