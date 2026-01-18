import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class Sampler(Metric):
    """A stateful class for updating a cumulative histogram metric.

  This class encapsulates a set of histograms (or a single histogram for a
  label-less metric) configured with a list of increasing bucket boundaries.
  Each histogram is identified by a tuple of labels. The class allows the
  user to add a sample to each histogram value.
  """
    __slots__ = []

    def __init__(self, name, buckets, description, *labels):
        """Creates a new Sampler.

    Args:
      name: name of the new metric.
      buckets: bucketing strategy of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
        super(Sampler, self).__init__('Sampler', _sampler_methods, len(labels), name, buckets.buckets, description, *labels)

    def get_cell(self, *labels):
        """Retrieves the cell."""
        return SamplerCell(super(Sampler, self).get_cell(*labels))