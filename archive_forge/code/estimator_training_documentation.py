import copy
import six
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
Function for evaluation.