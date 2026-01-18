from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@staticmethod
def _components_to_string(job, replica, task, device_type, device_index):
    """Stateless portion of `to_string` (separated to allow caching)."""
    key = (job, replica, task, device_type, device_index)
    cached_result = _COMPONENTS_TO_STRING_CACHE.get(key)
    if cached_result is not None:
        return cached_result
    output = []
    if job is not None:
        output.append('/job:' + job)
    if replica is not None:
        output.append('/replica:' + str(replica))
    if task is not None:
        output.append('/task:' + str(task))
    if device_type is not None:
        device_index_string = '*'
        if device_index is not None:
            device_index_string = str(device_index)
        output.append('/device:%s:%s' % (device_type, device_index_string))
    output = ''.join(output)
    _COMPONENTS_TO_STRING_CACHE[key] = output
    return output