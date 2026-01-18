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
def _ValidateFlags(self, flags: typing.Dict[str, str], runtimes: typing.Set[str]) -> None:
    if '--entry-point' not in flags:
        raise exceptions.RequiredArgumentException('--entry-point', 'Flag `--entry-point` required.')
    if '--builder' not in flags and '--runtime' not in flags:
        flags['--runtime'] = self._PromptUserForRuntime(runtimes)
    if flags.get('--runtime') not in runtimes:
        log.out.Print('--runtime must be one of the following:')
        flags['--runtime'] = self._PromptUserForRuntime(runtimes)