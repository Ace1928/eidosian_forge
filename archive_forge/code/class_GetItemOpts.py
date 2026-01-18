import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
class GetItemOpts(collections.namedtuple('GetItemOpts', ('element_dtype',))):
    pass