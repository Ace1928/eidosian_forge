from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateImageResourcePresentationSpec(group_help, image_prefix=''):
    name, flag_name_overrides = ('--image', {})
    if image_prefix:
        name = '--{}-image'.format(image_prefix)
        flag_name_overrides = {'project': '--{}-project'.format(image_prefix)}
    return presentation_specs.ResourcePresentationSpec(name, GetImageResourceSpec(), group_help=group_help, required=True, prefixes=False, flag_name_overrides=flag_name_overrides)