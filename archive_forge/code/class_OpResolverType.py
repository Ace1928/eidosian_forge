import ctypes
import enum
import os
import platform
import sys
import numpy as np
@_tf_export('lite.experimental.OpResolverType')
@enum.unique
class OpResolverType(enum.Enum):
    """Different types of op resolvers for Tensorflow Lite.

  * `AUTO`: Indicates the op resolver that is chosen by default in TfLite
     Python, which is the "BUILTIN" as described below.
  * `BUILTIN`: Indicates the op resolver for built-in ops with optimized kernel
    implementation.
  * `BUILTIN_REF`: Indicates the op resolver for built-in ops with reference
    kernel implementation. It's generally used for testing and debugging.
  * `BUILTIN_WITHOUT_DEFAULT_DELEGATES`: Indicates the op resolver for
    built-in ops with optimized kernel implementation, but it will disable
    the application of default TfLite delegates (like the XNNPACK delegate) to
    the model graph. Generally this should not be used unless there are issues
    with the default configuration.
  """
    AUTO = 0
    BUILTIN = 1
    BUILTIN_REF = 2
    BUILTIN_WITHOUT_DEFAULT_DELEGATES = 3