from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeAggregate(_messages.Message):
    """A value that combines incremental updates into a summarized value. Data
  is never directly written or read using type `Aggregate`. Writes will
  provide either the `input_type` or `state_type`, and reads will always
  return the `state_type` .

  Fields:
    inputType: Type of the inputs that are accumulated by this `Aggregate`,
      which must specify a full encoding. Use `AddInput` mutations to
      accumulate new inputs.
    stateType: Output only. Type that holds the internal accumulator state for
      the `Aggregate`. This is a function of the `input_type` and `aggregator`
      chosen, and will always specify a full encoding.
    sum: Sum aggregator.
  """
    inputType = _messages.MessageField('Type', 1)
    stateType = _messages.MessageField('Type', 2)
    sum = _messages.MessageField('GoogleBigtableAdminV2TypeAggregateSum', 3)