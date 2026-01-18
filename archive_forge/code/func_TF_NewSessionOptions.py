from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.python.client._pywrap_tf_session import _TF_SetTarget
from tensorflow.python.client._pywrap_tf_session import _TF_SetConfig
from tensorflow.python.client._pywrap_tf_session import _TF_NewSessionOptions
from tensorflow.python.util import tf_stack
def TF_NewSessionOptions(target=None, config=None):
    opts = _TF_NewSessionOptions()
    if target is not None:
        _TF_SetTarget(opts, target)
    if config is not None:
        config_str = config.SerializeToString()
        _TF_SetConfig(opts, config_str)
    return opts