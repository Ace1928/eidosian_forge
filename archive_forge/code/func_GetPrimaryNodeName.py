from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
import subprocess
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from six.moves import range
def GetPrimaryNodeName():
    """Get the primary node name.

  Returns:
    str, the name of the primary node. If running in tensorflow 1.x,
    return 'master'. If running in tensorflow 2.x, return 'chief'.
    If tensorflow is not installed in local envrionment, it will return
    the default name 'chief'.
  Raises:
    ValueError: if there is no python executable on the user system thrown by
      execution_utils.GetPythonExecutable.
  """
    exe_override = properties.VALUES.ml_engine.local_python.Get()
    python_executable = exe_override or files.FindExecutableOnPath('python') or execution_utils.GetPythonExecutable()
    cmd = [python_executable, '-c', 'import tensorflow as tf; print(tf.version.VERSION)']
    with files.FileWriter(os.devnull) as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=f)
    return_code = proc.wait()
    if return_code != 0:
        log.warning('\n    Cannot import tensorflow under path {}. Using "chief" for cluster setting.\n    If this is not intended, Please check if tensorflow is installed. Please also\n    verify if the python path used is correct. If not, to change the python path:\n    use `gcloud config set ml_engine/local_python $python_path`\n    Eg: gcloud config set ml_engine/local_python /usr/bin/python3'.format(python_executable))
        return 'chief'
    tf_version = proc.stdout.read()
    if 'decode' in dir(tf_version):
        tf_version = tf_version.decode('utf-8')
    if tf_version.startswith('1.'):
        return 'master'
    elif tf_version.startswith('2.'):
        return 'chief'
    log.warning('Unexpected tensorflow version {}, using the default primary node name, aka "chief" for cluster settings'.format(tf_version))
    return 'chief'