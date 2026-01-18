import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class SamplerCell(object):
    """SamplerCell stores each value of a Sampler."""
    __slots__ = ['_cell']

    def __init__(self, cell):
        """Creates a new SamplerCell.

    Args:
      cell: A c pointer of TFE_MonitoringSamplerCell.
    """
        self._cell = cell

    def add(self, value):
        """Atomically add a sample.

    Args:
      value: float value.
    """
        pywrap_tfe.TFE_MonitoringSamplerCellAdd(self._cell, value)

    def value(self):
        """Retrieves the current distribution of samples.

    Returns:
      A HistogramProto describing the distribution of samples.
    """
        with c_api_util.tf_buffer() as buffer_:
            pywrap_tfe.TFE_MonitoringSamplerCellValue(self._cell, buffer_)
            proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
        histogram_proto = summary_pb2.HistogramProto()
        histogram_proto.ParseFromString(compat.as_bytes(proto_data))
        return histogram_proto