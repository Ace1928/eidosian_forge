from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import device_filters_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _make_cluster_device_filters(self):
    """Creates `ClusterDeviceFilters` proto based on the `_device_filters`.

    Raises:
      TypeError: If `_device_filters` is not a dictionary mapping strings to
      a map of task indices and device filters.
    """
    self._cluster_device_filters = device_filters_pb2.ClusterDeviceFilters()
    for job_name, tasks in sorted(self._device_filters.items()):
        try:
            job_name = compat.as_bytes(job_name)
        except TypeError:
            raise TypeError('Job name %r must be bytes or unicode' % job_name)
        jdf = self._cluster_device_filters.jobs.add()
        jdf.name = job_name
        for i, task_device_filters in sorted(tasks.items()):
            for tdf in task_device_filters:
                try:
                    tdf = compat.as_bytes(tdf)
                except TypeError:
                    raise TypeError('Device filter %r must be bytes or unicode' % tdf)
                jdf.tasks[i].device_filters.append(tdf)