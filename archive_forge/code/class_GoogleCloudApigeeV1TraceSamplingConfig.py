from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TraceSamplingConfig(_messages.Message):
    """TraceSamplingConfig represents the detail settings of distributed
  tracing. Only the fields that are defined in the distributed trace
  configuration can be overridden using the distribute trace configuration
  override APIs.

  Enums:
    SamplerValueValuesEnum: Sampler of distributed tracing. OFF is the default
      value.

  Fields:
    sampler: Sampler of distributed tracing. OFF is the default value.
    samplingRate: Field sampling rate. This value is only applicable when
      using the PROBABILITY sampler. The supported values are > 0 and <= 0.5.
  """

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
    sampler = _messages.EnumField('SamplerValueValuesEnum', 1)
    samplingRate = _messages.FloatField(2, variant=_messages.Variant.FLOAT)