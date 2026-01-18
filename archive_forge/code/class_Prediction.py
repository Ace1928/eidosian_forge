from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class Prediction(base.Command):
    """Start an AI Platform batch prediction job."""

    @staticmethod
    def Args(parser):
        _AddSubmitPredictionArgs(parser)
        parser.display_info.AddFormat(jobs_util.JOB_FORMAT)

    def Run(self, args):
        data_format = jobs_util.DataFormatFlagMap().GetEnumForChoice(args.data_format)
        jobs_client = jobs.JobsClient()
        labels = jobs_util.ParseCreateLabels(jobs_client, args)
        return jobs_util.SubmitPrediction(jobs_client, args.job, model_dir=args.model_dir, model=args.model, version=args.version, input_paths=args.input_paths, data_format=data_format.name, output_path=args.output_path, region=args.region, runtime_version=args.runtime_version, max_worker_count=args.max_worker_count, batch_size=args.batch_size, signature_name=args.signature_name, labels=labels)