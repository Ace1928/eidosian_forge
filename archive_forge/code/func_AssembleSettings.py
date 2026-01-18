from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def AssembleSettings(args):
    settings = Settings.Defaults()
    context_dir = getattr(args, 'source', None) or os.path.curdir
    service_config = getattr(args, 'service_config', None)
    yaml_file = common.ChooseExistingServiceYaml(context_dir, service_config)
    if yaml_file:
        settings = settings.WithServiceYaml(yaml_file)
    settings = settings.WithArgs(args)
    return settings.Build()