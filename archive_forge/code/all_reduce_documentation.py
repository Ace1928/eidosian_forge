import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
Construct hybrid of Shuffle within workers, Shuffle across workers.