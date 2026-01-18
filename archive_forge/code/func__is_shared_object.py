import errno
import hashlib
import importlib
import os
import platform
import sys
from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _is_shared_object(filename):
    """Check the file to see if it is a shared object, only using extension."""
    if platform.system() == 'Linux':
        if filename.endswith('.so'):
            return True
        else:
            index = filename.rfind('.so.')
            if index == -1:
                return False
            else:
                return filename[index + 4].isdecimal()
    elif platform.system() == 'Darwin':
        return filename.endswith('.dylib')
    elif platform.system() == 'Windows':
        return filename.endswith('.dll')
    else:
        return False