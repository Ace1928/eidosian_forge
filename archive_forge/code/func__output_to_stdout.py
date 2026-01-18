import os
import sys
import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _output_to_stdout(msg, *args, **kwargs):
    print(msg % args)
    if kwargs.get('exc_info', False):
        traceback.print_exc()