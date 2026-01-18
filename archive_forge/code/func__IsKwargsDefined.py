from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _IsKwargsDefined(key, **kwargs):
    return key in kwargs and bool(kwargs.get(key))