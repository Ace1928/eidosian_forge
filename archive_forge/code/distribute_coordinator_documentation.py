import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
A fake server that runs a master session.