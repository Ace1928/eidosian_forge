from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def _ParseCAResourceArgs(args):
    """Parses, validates and returns the resource args from the CLI.

  Args:
    args: The parsed arguments from the command-line.

  Returns:
    Tuple containing the Resource objects for (CA, source CA, issuer).
  """
    resource_args.ValidateResourceIsCompleteIfSpecified(args, 'kms_key_version')
    resource_args.ValidateResourceIsCompleteIfSpecified(args, 'issuer_pool')
    resource_args.ValidateResourceIsCompleteIfSpecified(args, 'from_ca')
    ca_ref = args.CONCEPTS.certificate_authority.Parse()
    resource_args.ValidateResourceLocation(ca_ref, 'CERTIFICATE_AUTHORITY', version='v1')
    kms_key_version_ref = args.CONCEPTS.kms_key_version.Parse()
    if kms_key_version_ref and ca_ref.locationsId != kms_key_version_ref.locationsId:
        raise exceptions.InvalidArgumentException('--kms-key-version', 'KMS key must be in the same location as the Certificate Authority ({}).'.format(ca_ref.locationsId))
    issuer_ref = args.CONCEPTS.issuer_pool.Parse() if hasattr(args, 'issuer_pool') else None
    source_ca_ref = args.CONCEPTS.from_ca.Parse()
    if source_ca_ref and source_ca_ref.Parent().RelativeName() != ca_ref.Parent().RelativeName():
        raise exceptions.InvalidArgumentException('--from-ca', 'The provided source CA must be a part of the same pool as the specified CA to be created.')
    return (ca_ref, source_ca_ref, issuer_ref)