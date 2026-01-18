from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SamplerValueValuesEnum(_messages.Enum):
    """Sampler of distributed tracing. OFF is the default value.

    Values:
      SAMPLER_UNSPECIFIED: Sampler unspecified.
      OFF: OFF means distributed trace is disabled, or the sampling
        probability is 0.
      PROBABILITY: PROBABILITY means traces are captured on a probability that
        defined by sampling_rate. The sampling rate is limited to 0 to 0.5
        when this is set.
    """
    SAMPLER_UNSPECIFIED = 0
    OFF = 1
    PROBABILITY = 2