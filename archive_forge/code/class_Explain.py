from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import predict
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.command_lib.ml_engine import region_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Explain(base.Command):
    """Run AI Platform explanation.

     `{command}` sends an explain request to AI Platform for the given
     instances. This command will read up to 100 instances, though the service
     itself will accept instances up to the payload limit size (currently,
     1.5MB).
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        parser.add_argument('--model', required=True, help='Name of the model.')
        flags.GetRegionArg(include_global=True).AddToParser(parser)
        parser.add_argument('--version', help='Model version to be used.\n\nIf unspecified, the default version of the model will be used. To list model\nversions run\n\n  $ {parent_command} versions list\n')
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--json-request', help='        Path to a local file containing the body of JSON request.\n\n        An example of a JSON request:\n\n            {\n              "instances": [\n                {"x": [1, 2], "y": [3, 4]},\n                {"x": [-1, -2], "y": [-3, -4]}\n              ]\n            }\n\n        This flag accepts "-" for stdin.\n        ')
        group.add_argument('--json-instances', help='        Path to a local file from which instances are read.\n        Instances are in JSON format; newline delimited.\n\n        An example of the JSON instances file:\n\n            {"images": [0.0, ..., 0.1], "key": 3}\n            {"images": [0.0, ..., 0.1], "key": 2}\n            ...\n\n        This flag accepts "-" for stdin.\n        ')
        group.add_argument('--text-instances', help='        Path to a local file from which instances are read.\n        Instances are in UTF-8 encoded text format; newline delimited.\n\n        An example of the text instances file:\n\n            107,4.9,2.5,4.5,1.7\n            100,5.7,2.8,4.1,1.3\n            ...\n\n        This flag accepts "-" for stdin.\n        ')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        instances = predict_utilities.ReadInstancesFromArgs(args.json_request, args.json_instances, args.text_instances, limit=INPUT_INSTANCES_LIMIT)
        region = region_util.GetRegion(args)
        with endpoint_util.MlEndpointOverrides(region=region):
            model_or_version_ref = predict_utilities.ParseModelOrVersionRef(args.model, args.version)
            results = predict.Explain(model_or_version_ref, instances)
        if not args.IsSpecified('format'):
            args.format = predict_utilities.GetDefaultFormat(results.get('predictions'))
        return results