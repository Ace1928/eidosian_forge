import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _log(self, message):
    """Log and print authoring warning / error message."""
    self._log_messages.append(message)
    print(message)