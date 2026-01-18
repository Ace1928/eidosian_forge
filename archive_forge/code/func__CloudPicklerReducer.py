import ray
import ray.cloudpickle as pickle
from ray.util.annotations import DeveloperAPI, PublicAPI
def _CloudPicklerReducer(obj):
    return (custom_deserializer, (custom_serializer(obj),))