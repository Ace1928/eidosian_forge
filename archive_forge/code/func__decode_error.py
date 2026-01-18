import functools
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _decode_error(self, err):
    """Parses the given ConverterError and generates compatibility warnings."""
    if hasattr(err, 'errors'):
        self._decode_converter_error(err)
    else:
        self._decode_error_legacy(err)
    if self._raise_exception and self._log_messages:
        raise CompatibilityError(f'CompatibilityException at {repr(self._func)}')