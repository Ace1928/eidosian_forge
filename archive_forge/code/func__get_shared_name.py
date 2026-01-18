import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
def _get_shared_name():
    global _shared_name_counter
    with _module_lock:
        val = _shared_name_counter
        _shared_name_counter += 1
    return 'c%s' % val