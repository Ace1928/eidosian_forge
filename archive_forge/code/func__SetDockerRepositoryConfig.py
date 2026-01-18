import re
import types
from typing import FrozenSet, Optional, Tuple
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.functions.v2 import types as api_types
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.eventarc import types as trigger_types
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import labels_util
from googlecloudsdk.command_lib.functions import run_util
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.command_lib.functions.v2 import deploy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
def _SetDockerRepositoryConfig(args: parser_extensions.Namespace, function: api_types.Function, existing_function: Optional[api_types.Function], function_ref: resources.Resource) -> FrozenSet[str]:
    """Sets user-provided docker repository field on the function.

  Args:
    args: arguments that this command was invoked with
    function: `cloudfunctions_v2_messages.Function`, recently created or updated
      GCF function.
    existing_function: `cloudfunctions_v2_messages.Function | None`,
      pre-existing function.
    function_ref: resource reference.

  Returns:
    A set of update mask fields.
  """
    updated_fields = set()
    function.buildConfig.dockerRepository = existing_function.buildConfig.dockerRepository if existing_function else None
    if args.IsSpecified('docker_repository'):
        cmek_util.ValidateDockerRepositoryForFunction(args.docker_repository, function_ref)
    if args.IsSpecified('docker_repository') or args.IsSpecified('clear_docker_repository'):
        updated_docker_repository = None if args.IsSpecified('clear_docker_repository') else args.docker_repository
        function.buildConfig.dockerRepository = cmek_util.NormalizeDockerRepositoryFormat(updated_docker_repository)
        if existing_function is None or function.buildConfig.dockerRepository != existing_function.buildConfig.dockerRepository:
            updated_fields.add('build_config.docker_repository')
    if function.kmsKeyName and (not function.buildConfig.dockerRepository):
        raise calliope_exceptions.RequiredArgumentException('--docker-repository', 'A Docker repository must be specified when a KMS key is configured for the function.')
    return updated_fields