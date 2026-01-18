import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
Returns the name of the TPU device assigned to a logical core.