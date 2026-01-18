import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
def _bns_task_id(job: str) -> Union[int, str]:
    """Tries to extract an integer task ID from a job name.

  For example, for `job` = '/.../tpu_worker/0:port_name', return 0.

  Args:
    job: A job name to extract task ID from.

  Returns:
    The task ID on success, or the original job name on failure.
  """
    maybe_task_id = job.rsplit('/')[-1].rsplit(':')[0]
    try:
        return int(maybe_task_id)
    except ValueError:
        return job