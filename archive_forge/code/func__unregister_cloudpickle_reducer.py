import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
def _unregister_cloudpickle_reducer(self, cls):
    pickle.CloudPickler.dispatch.pop(cls, None)