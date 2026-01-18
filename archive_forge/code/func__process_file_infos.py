import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _process_file_infos(file_infos):
    """Computes aggregate information about files to read.

  The method collects information about the files to read, the total number of
  elements, and arrays that can be used to account for elements to be skipped,
  which can be specified via the "skip" and "take" keys.

  To account for elements to skip, the range of each file can be divided into
  three regions:
  - S (elements to skip)
  - T (elements to read)
  - R (remainder of elements that will also be skipped)

  The `thresholds` and `offsets` arrays are initialized as follows:
  `thresholds = [0, T_1, T_1 + T_2, ...]` and
  `offsets = [S_1, S_1 + R_1 + S_2, S_1 + R_1 + S_2 + R_2 + S_3, ...]`

  This makes it possible to map an index from a contiguous range
  `(0...num_elements_to_read)` to an index in the range of all elements,
  skipping over elements as per the "skip" and "take" keys values. In
  particular, for a given input index `X`, we find the greatest `thresholds`
  value that is smaller or equal to `X`. Let `t(X)` denotes such index in the
  `thresholds` array. The output index is computed as `X + offsets[t(X)]`.

  Args:
    file_infos: See `file_infos` argument of `index_shuffle` for details.

  Returns:
    A dictionary containing the following keys:
      - `files`, the vector of pathnames of files to read
      - `num_elements`, an integer identifying the total number of elements
      - `offsets`, the vector of offsets to use for index adjustment (in case
        any elements should be skipped)
      - `thresholds`, the vector of thresholds to use for index adjustment (in
        case any elements should be skipped)
  """
    files = []
    num_elements = 0
    offsets = np.int64([])
    offset_sum = 0
    thresholds = np.int64([])
    threshold_sum = 0
    adjustment_needed = False
    for file_info in file_infos:
        files.append(file_info['path'])
        skip = 0
        if 'skip' in file_info:
            if file_info['skip'] < -1:
                raise ValueError('`skip` should be greater than `-1` but got {}'.format(file_info['skip']))
            if file_info['skip'] == -1:
                skip = file_info['num_elements']
            else:
                skip = min(file_info['skip'], file_info['num_elements'])
        take = file_info['num_elements'] - skip
        if 'take' in file_info:
            if file_info['take'] < -1:
                raise ValueError('`take` should be greater than `-1` but got {}'.format(file_info['take']))
            if file_info['take'] != -1:
                take = min(file_info['take'], take)
        remainder = file_info['num_elements'] - skip - take
        if take != file_info['num_elements']:
            adjustment_needed = True
        num_elements += take
        offsets = np.append(offsets, offset_sum + skip)
        offset_sum += skip + remainder
        thresholds = np.append(thresholds, threshold_sum)
        threshold_sum += take
    result = {'files': files, 'num_elements': num_elements}
    if adjustment_needed:
        result['offsets'] = offsets
        result['thresholds'] = thresholds
    return result