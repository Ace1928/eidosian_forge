from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ThreatPreventionProfile(_messages.Message):
    """ThreatPreventionProfile defines an action for specific threat signatures
  or severity levels.

  Fields:
    severityOverrides: Optional. Configuration for overriding threats actions
      by severity match.
    threatOverrides: Optional. Configuration for overriding threats actions by
      threat_id match. If a threat is matched both by configuration provided
      in severity_overrides and threat_overrides, the threat_overrides action
      is applied.
  """
    severityOverrides = _messages.MessageField('SeverityOverride', 1, repeated=True)
    threatOverrides = _messages.MessageField('ThreatOverride', 2, repeated=True)