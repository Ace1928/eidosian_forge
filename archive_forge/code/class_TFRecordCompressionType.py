from tensorflow.python.lib.io import _pywrap_record_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['io.TFRecordCompressionType', 'python_io.TFRecordCompressionType'])
@deprecation.deprecated_endpoints('io.TFRecordCompressionType', 'python_io.TFRecordCompressionType')
class TFRecordCompressionType(object):
    """The type of compression for the record."""
    NONE = 0
    ZLIB = 1
    GZIP = 2