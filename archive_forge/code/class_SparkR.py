from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc.batches import batch_submitter
from googlecloudsdk.command_lib.dataproc.batches import sparkr_batch_factory
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SparkR(base.Command):
    """Submit a Spark R batch job."""
    detailed_help = {'EXAMPLES': '          To submit a Spark R batch job running "my-main-r.r" script and upload it to "gs://my-bucket", run:\n\n            $ {command} my-main-r.r --deps-bucket=gs://my-bucket --region=\'us-central1\' -- ARG1 ARG2\n          '}

    @staticmethod
    def Args(parser):
        sparkr_batch_factory.AddArguments(parser)

    def Run(self, args):
        dataproc = dp.Dataproc(base.ReleaseTrack.GA)
        sparkr_batch = sparkr_batch_factory.SparkRBatchFactory(dataproc).UploadLocalFilesAndGetMessage(args)
        return batch_submitter.Submit(sparkr_batch, dataproc, args)