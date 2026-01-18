import itertools
from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging
def _dedup_strings(device_strs):
    """Groups together consecutive identical strings.

  For example, given:
      ['GPU 1', 'GPU 2', 'GPU 2', 'GPU 3', 'GPU 3', 'GPU 3']
  This function returns:
      ['GPU 1', 'GPU 2 (x2)', 'GPU 3 (x3)']

  Args:
    device_strs: A list of strings, each representing a device.

  Returns:
    A copy of the input, but identical consecutive strings are merged into a
    single string.
  """
    new_device_strs = []
    for device_str, vals in itertools.groupby(device_strs):
        num = len(list(vals))
        if num == 1:
            new_device_strs.append(device_str)
        else:
            new_device_strs.append('%s (x%d)' % (device_str, num))
    return new_device_strs