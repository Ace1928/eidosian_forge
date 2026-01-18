import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
class CallableObjectWrapper(CloudpickledObjectWrapper):

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)