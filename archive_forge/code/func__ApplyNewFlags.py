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
def _ApplyNewFlags(self, args: parser_extensions.Namespace, old_flags: typing.Dict[str, str]) -> typing.Dict[str, str]:
    flags = {**old_flags, **args.GetSpecifiedArgs()}
    flags = {k: v for k, v in flags.items() if not ('NAME' in k or 'env-vars' in k)}
    return flags