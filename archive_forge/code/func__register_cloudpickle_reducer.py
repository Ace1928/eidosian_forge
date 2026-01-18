import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
def _register_cloudpickle_reducer(self, cls, reducer):
    pickle.CloudPickler.dispatch[cls] = reducer