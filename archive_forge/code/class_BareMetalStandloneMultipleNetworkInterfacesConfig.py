from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandloneMultipleNetworkInterfacesConfig(_messages.Message):
    """Specifies the multiple networking interfaces cluster configuration.

  Fields:
    enabled: Whether to enable multiple network interfaces for your pods. When
      set network_config.advanced_networking is automatically set to true.
  """
    enabled = _messages.BooleanField(1)