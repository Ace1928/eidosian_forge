from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseExportPolicy(self, volume, export_policy):
    """Parses Export Policy for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object
      export_policy: the Export Policy message object.

    Returns:
      Volume message populated with Export Policy values.

    """
    if not export_policy:
        return
    export_policy_config = self.messages.ExportPolicy()
    for policy in export_policy:
        simple_export_policy_rule = self.messages.SimpleExportPolicyRule()
        for key, val in policy.items():
            if key == 'allowed-clients':
                simple_export_policy_rule.allowedClients = val
            if key == 'access-type':
                simple_export_policy_rule.accessType = self.messages.SimpleExportPolicyRule.AccessTypeValueValuesEnum.lookup_by_name(val)
            if key == 'has-root-access':
                simple_export_policy_rule.hasRootAccess = val
            if key == 'kerberos-5-read-only':
                simple_export_policy_rule.kerberos5ReadOnly = val
            if key == 'kerberos-5-read-write':
                simple_export_policy_rule.kerberos5ReadWrite = val
            if key == 'kerberos-5i-read-only':
                simple_export_policy_rule.kerberos5iReadOnly = val
            if key == 'kerberos-5i-read-write':
                simple_export_policy_rule.kerberos5iReadWrite = val
            if key == 'kerberos-5p-read-only':
                simple_export_policy_rule.kerberos5pReadOnly = val
            if key == 'kerberos-5p-read-write':
                simple_export_policy_rule.kerberos5pReadWrite = val
            if key == 'nfsv3':
                simple_export_policy_rule.nfsv3 = val
            if key == 'nfsv4':
                simple_export_policy_rule.nfsv4 = val
        export_policy_config.rules.append(simple_export_policy_rule)
    volume.exportPolicy = export_policy_config