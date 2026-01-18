from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc.batches import batch_submitter
from googlecloudsdk.command_lib.dataproc.batches import pyspark_batch_factory
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class PySpark(base.Command):
    """Submit a PySpark batch job."""
    detailed_help = {'EXAMPLES': '          To submit a PySpark batch job called "my-batch" that runs "my-pyspark.py", run:\n\n            $ {command} my-pyspark.py --batch=my-batch --deps-bucket=gs://my-bucket --region=us-central1 --py-files=\'path/to/my/python/script.py\'\n          '}

    @staticmethod
    def Args(parser):
        pyspark_batch_factory.AddArguments(parser)

    def Run(self, args):
        dataproc = dp.Dataproc(base.ReleaseTrack.GA)
        pyspark_batch = pyspark_batch_factory.PySparkBatchFactory(dataproc).UploadLocalFilesAndGetMessage(args)
        return batch_submitter.Submit(pyspark_batch, dataproc, args)