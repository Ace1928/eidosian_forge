from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import local_utils
from googlecloudsdk.command_lib.ml_engine import predict_utilities
from googlecloudsdk.core import log
class Predict(base.Command):
    """Run prediction locally."""

    @staticmethod
    def Args(parser):
        _AddLocalPredictArgs(parser)

    def Run(self, args):
        framework = flags.FRAMEWORK_MAPPER.GetEnumForChoice(args.framework)
        framework_flag = framework.name.lower() if framework else 'tensorflow'
        if args.signature_name is None:
            log.status.Print('If the signature defined in the model is not serving_default then you must specify it via --signature-name flag, otherwise the command may fail.')
        results = local_utils.RunPredict(args.model_dir, json_request=args.json_request, json_instances=args.json_instances, text_instances=args.text_instances, framework=framework_flag, signature_name=args.signature_name)
        if not args.IsSpecified('format'):
            if isinstance(results, list):
                predictions = results
            else:
                predictions = results.get('predictions')
            args.format = predict_utilities.GetDefaultFormat(predictions)
        return results