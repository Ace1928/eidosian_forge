from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _ns_msgs(self, ns):
    """Builds the PolicyControllerBundleInstallSpec from namespace list."""
    install_spec = self.messages.PolicyControllerBundleInstallSpec()
    install_spec.exemptedNamespaces = ns
    return install_spec