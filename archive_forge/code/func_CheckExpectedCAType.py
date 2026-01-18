from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.kms import resource_args as kms_args
from googlecloudsdk.command_lib.privateca import completers as privateca_completers
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def CheckExpectedCAType(expected_type, ca, version='v1'):
    """Raises an exception if the Certificate Authority type is not expected_type.

  Args:
    expected_type: The expected type.
    ca: The ca object to check.
    version: The version of the API to check against.
  """
    ca_type_enum = base.GetMessagesModule(api_version=version).CertificateAuthority.TypeValueValuesEnum
    if expected_type == ca_type_enum.SUBORDINATE and ca.type != expected_type:
        raise privateca_exceptions.InvalidCertificateAuthorityTypeError('Cannot perform subordinates command on Root CA. Please use the `privateca roots` command group instead.')
    elif expected_type == ca_type_enum.SELF_SIGNED and ca.type != expected_type:
        raise privateca_exceptions.InvalidCertificateAuthorityTypeError('Cannot perform roots command on Subordinate CA. Please use the `privateca subordinates` command group instead.')