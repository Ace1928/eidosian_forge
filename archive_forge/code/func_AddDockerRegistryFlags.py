from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddDockerRegistryFlags(parser):
    """Adds flags for selecting the Docker registry type for Cloud Function."""
    docker_registry_arg = base.ChoiceArgument('--docker-registry', choices=sorted(DOCKER_REGISTRY_MAPPING.values()), help_str="        Docker Registry to use for storing the function's Docker images.\n        The option `artifact-registry` is used by default.\n\n        Warning: Artifact Registry and Container Registry have different image\n        storage costs. For more details, please see\n        https://cloud.google.com/functions/pricing#deployment_costs\n      ")
    docker_registry_arg.AddToParser(parser)