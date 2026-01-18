from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyController(_messages.Message):
    """Configuration for Policy Controller

  Fields:
    auditIntervalSeconds: Sets the interval for Policy Controller Audit Scans
      (in seconds). When set to 0, this disables audit functionality
      altogether.
    enabled: Enables the installation of Policy Controller. If false, the rest
      of PolicyController fields take no effect.
    exemptableNamespaces: The set of namespaces that are excluded from Policy
      Controller checks. Namespaces do not need to currently exist on the
      cluster.
    logDeniesEnabled: Logs all denies and dry run failures.
    monitoring: Monitoring specifies the configuration of monitoring.
    mutationEnabled: Enable users to try out mutation for PolicyController.
    referentialRulesEnabled: Enables the ability to use Constraint Templates
      that reference to objects other than the object currently being
      evaluated.
    templateLibraryInstalled: Installs the default template library along with
      Policy Controller.
    updateTime: Output only. Last time this membership spec was updated.
  """
    auditIntervalSeconds = _messages.IntegerField(1)
    enabled = _messages.BooleanField(2)
    exemptableNamespaces = _messages.StringField(3, repeated=True)
    logDeniesEnabled = _messages.BooleanField(4)
    monitoring = _messages.MessageField('PolicyControllerMonitoring', 5)
    mutationEnabled = _messages.BooleanField(6)
    referentialRulesEnabled = _messages.BooleanField(7)
    templateLibraryInstalled = _messages.BooleanField(8)
    updateTime = _messages.StringField(9)