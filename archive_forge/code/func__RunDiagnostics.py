from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.diagnostics import network_diagnostics
from googlecloudsdk.core.diagnostics import property_diagnostics
def _RunDiagnostics(ignore_hidden_property_allowlist):
    passed_network = network_diagnostics.NetworkDiagnostic().RunChecks()
    passed_props = property_diagnostics.PropertyDiagnostic(ignore_hidden_property_allowlist).RunChecks()
    return passed_network and passed_props