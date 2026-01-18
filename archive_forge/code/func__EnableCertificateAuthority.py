from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.privateca import create_utils
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import iam
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import p4sa
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.privateca import storage
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _EnableCertificateAuthority(self, ca_name):
    """Enables the given CA."""
    enable_request = self.messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest(name=ca_name, enableCertificateAuthorityRequest=self.messages.EnableCertificateAuthorityRequest(requestId=request_utils.GenerateRequestId()))
    operation = self.client.projects_locations_caPools_certificateAuthorities.Enable(enable_request)
    return operations.Await(operation, 'Enabling CA.', api_version='v1')