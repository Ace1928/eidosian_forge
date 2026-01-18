from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScanConfig(_messages.Message):
    """Indicates various scans and whether they are turned on or off.

  Fields:
    createTime: Output only. The time this scan config was created.
    description: Output only. A human-readable description of what the
      `ScanConfig` does.
    enabled: Indicates whether the Scan is enabled.
    name: Output only. The name of the ScanConfig in the form
      "projects/{project_id}/scanConfigs/{scan_config_id}".
    updateTime: Output only. The time this scan config was last updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    enabled = _messages.BooleanField(3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)