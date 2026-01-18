from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseProjectConfigWithPushblock(args):
    project_ref = CreateProjectResource(args)
    project_name = project_ref.RelativeName()
    enable_pushblock = args.enable_pushblock
    return _MESSAGES.ProjectConfig(enablePrivateKeyCheck=enable_pushblock, name=project_name, pubsubConfigs=None)