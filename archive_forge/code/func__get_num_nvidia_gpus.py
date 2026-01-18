import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _get_num_nvidia_gpus():
    """Gets the number of NVIDIA GPUs by using CUDA_VISIBLE_DEVICES and nvidia-smi.

  Returns:
    Number of GPUs available on the node
  Raises:
    RuntimeError if executing nvidia-smi failed
  """
    try:
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except KeyError:
        pass
    try:
        output = subprocess.check_output(['nvidia-smi', '--list-gpus'], encoding='utf-8')
        return sum((l.startswith('GPU ') for l in output.strip().split('\n')))
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Could not get number of GPUs from nvidia-smi. Maybe it is missing?\nOutput: %s' % e.output)