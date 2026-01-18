from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUploadModelFlags(parser, prompt_func=region_util.PromptForRegion):
    """Adds flags for UploadModel.

  Args:
    parser: the parser for the command.
    prompt_func: function, the function to prompt for region from list of
      available regions which returns a string for the region selected. Default
      is region_util.PromptForRegion which contains three regions,
      'us-central1', 'europe-west4', and 'asia-east1'.
  """
    AddRegionResourceArg(parser, 'to upload model', prompt_func=prompt_func)
    base.Argument('--display-name', required=True, help='Display name of the model.').AddToParser(parser)
    base.Argument('--description', required=False, help='Description of the model.').AddToParser(parser)
    base.Argument('--version-description', required=False, help='Description of the model version.').AddToParser(parser)
    base.Argument('--container-image-uri', required=True, help='URI of the Model serving container file in the Container Registry\n(e.g. gcr.io/myproject/server:latest).\n').AddToParser(parser)
    base.Argument('--artifact-uri', help='Path to the directory containing the Model artifact and any of its\nsupporting files.\n').AddToParser(parser)
    parser.add_argument('--container-env-vars', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='List of key-value pairs to set as environment variables.')
    parser.add_argument('--container-command', type=arg_parsers.ArgList(), metavar='COMMAND', action=arg_parsers.UpdateAction, help="Entrypoint for the container image. If not specified, the container\nimage's default entrypoint is run.\n")
    parser.add_argument('--container-args', metavar='ARG', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help="Comma-separated arguments passed to the command run by the container\nimage. If not specified and no `--command` is provided, the container\nimage's default command is used.\n")
    parser.add_argument('--container-ports', metavar='PORT', type=arg_parsers.ArgList(element_type=arg_parsers.BoundedInt(1, 65535)), action=arg_parsers.UpdateAction, help='Container ports to receive http requests at. Must be a number between 1 and\n65535, inclusive.\n')
    parser.add_argument('--container-grpc-ports', metavar='PORT', type=arg_parsers.ArgList(element_type=arg_parsers.BoundedInt(1, 65535)), action=arg_parsers.UpdateAction, help='Container ports to receive grpc requests at. Must be a number between 1 and\n65535, inclusive.\n')
    parser.add_argument('--container-predict-route', help='HTTP path to send prediction requests to inside the container.')
    parser.add_argument('--container-health-route', help='HTTP path to send health checks to inside the container.')
    parser.add_argument('--container-deployment-timeout-seconds', type=int, help='Deployment timeout in seconds.')
    parser.add_argument('--container-shared-memory-size-mb', type=int, help='The amount of the VM memory to reserve as the shared memory for the model in\nmegabytes.\n  ')
    parser.add_argument('--container-startup-probe-exec', type=arg_parsers.ArgList(), metavar='STARTUP_PROBE_EXEC', help='Exec specifies the action to take. Used by startup probe. An example of this\nargument would be ["cat", "/tmp/healthy"].\n  ')
    parser.add_argument('--container-startup-probe-period-seconds', type=int, help='How often (in seconds) to perform the startup probe. Default to 10 seconds.\nMinimum value is 1.\n  ')
    parser.add_argument('--container-startup-probe-timeout-seconds', type=int, help='Number of seconds after which the startup probe times out. Defaults to 1 second.\nMinimum value is 1.\n  ')
    parser.add_argument('--container-health-probe-exec', type=arg_parsers.ArgList(), metavar='HEALTH_PROBE_EXEC', help='Exec specifies the action to take. Used by health probe. An example of this\nargument would be ["cat", "/tmp/healthy"].\n  ')
    parser.add_argument('--container-health-probe-period-seconds', type=int, help='How often (in seconds) to perform the health probe. Default to 10 seconds.\nMinimum value is 1.\n  ')
    parser.add_argument('--container-health-probe-timeout-seconds', type=int, help='Number of seconds after which the health probe times out. Defaults to 1 second.\nMinimum value is 1.\n  ')
    parser.add_argument('--explanation-method', help='Method used for explanation. Accepted values are `integrated-gradients`, `xrai` and `sampled-shapley`.')
    parser.add_argument('--explanation-metadata-file', help="Path to a local JSON file that contains the metadata describing the Model's input and output for explanation.")
    parser.add_argument('--explanation-step-count', type=int, help='Number of steps to approximate the path integral for explanation.')
    parser.add_argument('--explanation-path-count', type=int, help='Number of feature permutations to consider when approximating the Shapley values for explanation.')
    parser.add_argument('--smooth-grad-noisy-sample-count', type=int, help='Number of gradient samples used for approximation at explanation. Only applicable to explanation method `integrated-gradients` or `xrai`.')
    parser.add_argument('--smooth-grad-noise-sigma', type=float, help='Single float value used to add noise to all the features for explanation. Only applicable to explanation method `integrated-gradients` or `xrai`.')
    parser.add_argument('--smooth-grad-noise-sigma-by-feature', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='Noise sigma by features for explanation. Noise sigma represents the standard deviation of the gaussian kernel that will be used to add noise to interpolated inputs prior to computing gradients. Only applicable to explanation method `integrated-gradients` or `xrai`.')
    parser.add_argument('--parent-model', type=str, help="Resource name of the model into which to upload the version. Only specify this field when uploading a new version.\n\nValue should be provided in format: projects/``PROJECT_ID''/locations/``REGION''/models/``PARENT_MODEL_ID''\n")
    parser.add_argument('--model-id', type=str, help='ID to use for the uploaded Model, which will become the final component of the model resource name.')
    parser.add_argument('--version-aliases', metavar='VERSION_ALIASES', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, help='Aliases used to reference a model version instead of auto-generated version ID. The aliases mentioned in the flag will replace the aliases set in the model.')
    parser.add_argument('--labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, help='Labels with user-defined metadata to organize your Models.\n\nLabel keys and values can be no longer than 64 characters\n(Unicode codepoints), can only contain lowercase letters, numeric\ncharacters, underscores and dashes. International characters are allowed.\n\nSee https://goo.gl/xmQnxf for more information and examples of labels.\n')