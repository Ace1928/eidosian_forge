from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemAddonsConfig(_messages.Message):
    """Config that customers are allowed to define for GDCE system add-ons.

  Fields:
    ingress: Optional. Config for Ingress.
    sdsOperator: Optional. Config for SDS Operator.
    unmanagedKafkaConfig: Optional. Config for unmanaged Kafka.
  """
    ingress = _messages.MessageField('Ingress', 1)
    sdsOperator = _messages.MessageField('SdsOperator', 2)
    unmanagedKafkaConfig = _messages.MessageField('UnmanagedKafkaConfig', 3)