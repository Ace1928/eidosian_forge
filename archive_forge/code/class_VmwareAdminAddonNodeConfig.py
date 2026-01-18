from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminAddonNodeConfig(_messages.Message):
    """VmwareAdminAddonNodeConfig contains add-on node configurations for
  VMware admin cluster.

  Fields:
    autoResizeConfig: VmwareAutoResizeConfig config specifies auto resize
      config.
  """
    autoResizeConfig = _messages.MessageField('VmwareAutoResizeConfig', 1)