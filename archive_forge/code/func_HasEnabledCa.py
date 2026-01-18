from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def HasEnabledCa(ca_list, messages):
    """Checks if there are any enabled CAs in the CA list."""
    for ca in ca_list:
        if ca.state == messages.CertificateAuthority.StateValueValuesEnum.ENABLED:
            return True
    return False