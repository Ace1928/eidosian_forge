from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetKmsKeyPresentationSpec(resource, region_fallthrough=False, flag_overrides=None, permission_info=None):
    """Return a Presentation Spec for kms key resource argument.

  Args:
    resource: str, the name of the resource that the cryptokey will be used to
    protect.
    region_fallthrough: bool, True if the command has a region flag that should
      be used as a fallthrough for the kms location.
    flag_overrides: dict, The default flag names are 'kms-key', 'kms-keyring',
      'kms-location' and 'kms-project'. You can pass a dict of overrides where
      the keys of the dict are the default flag names, and the values are the
      override names.
    permission_info: str, optional permission info that overrides default
      permission info group help.

  Returns:
    Presentation spec suitable for adding to concept parser.
  """
    if not permission_info:
        permission_info = '{} must hold permission {}'.format("The 'Compute Engine Service Agent' service account", "'Cloud KMS CryptoKey Encrypter/Decrypter'")
    group_help = 'The Cloud KMS (Key Management Service) cryptokey that will be used to protect the {}. {}.'.format(resource, permission_info)
    presentation_spec = presentation_specs.ResourcePresentationSpec('--kms-key', GetKmsKeyResourceSpec(region_fallthrough=region_fallthrough), group_help, flag_name_overrides=flag_overrides or {})
    return presentation_spec