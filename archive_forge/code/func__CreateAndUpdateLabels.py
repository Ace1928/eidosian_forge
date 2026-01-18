from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
import typing
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags as flag_util
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.local import flags as local_flags
from googlecloudsdk.command_lib.functions.local import util
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def _CreateAndUpdateLabels(self, args: parser_extensions.Namespace) -> typing.Dict[str, typing.Any]:
    labels = {}
    old_labels = util.GetDockerContainerLabels(args.NAME[0])
    old_flags = json.loads(old_labels.get('flags', '{}'))
    old_env_vars = json.loads(old_labels.get('env-vars', '{}'))
    old_build_env_vars = json.loads(old_labels.get('build-env-vars', '{}'))
    labels['flags'] = self._ApplyNewFlags(args, old_flags)
    env_vars = map_util.GetMapFlagsFromArgs('env-vars', args)
    labels['env-vars'] = map_util.ApplyMapFlags(old_env_vars, **env_vars)
    build_env_vars = map_util.GetMapFlagsFromArgs('build-env-vars', args)
    labels['build-env-vars'] = map_util.ApplyMapFlags(old_build_env_vars, **build_env_vars)
    return labels