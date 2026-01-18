import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
Add a default serving output to the export_outputs if not present.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict.

  Returns:
    export_outputs dict with default serving signature added if necessary

  Raises:
    ValueError: if multiple export_outputs were provided without a default
      serving key.
  