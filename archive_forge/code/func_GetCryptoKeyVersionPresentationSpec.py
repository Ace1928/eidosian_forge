from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def GetCryptoKeyVersionPresentationSpec(group_help, base_name='keyversion', required=True, positional=True, use_global_project_flag=True):
    """Construct a resource spec for a CryptoKeyVersion flag."""
    flag_overrides = None
    if not use_global_project_flag:
        flag_overrides = {'project': _FormatArgName('{}-project'.format(base_name), positional)}
    return presentation_specs_lib.ResourcePresentationSpec(name=_FormatArgName(base_name, positional), concept_spec=_GetCryptoKeyVersionResourceSpec(), group_help=group_help, required=required, prefixes=not use_global_project_flag, flag_name_overrides=flag_overrides)