from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc.batches import batch_submitter
from googlecloudsdk.command_lib.dataproc.batches import sparksql_batch_factory
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SparkSql(base.Command):
    """Submit a Spark SQL batch job."""
    detailed_help = {'EXAMPLES': '          To submit a Spark SQL job running "my-sql-script.sql" and upload it to "gs://my-bucket", run:\n\n            $ {command} my-sql-script.sql --deps-bucket=gs://my-bucket --region=us-central1 --vars="NAME=VALUE,NAME2=VALUE2"\n          '}

    @staticmethod
    def Args(parser):
        sparksql_batch_factory.AddArguments(parser)

    def Run(self, args):
        dataproc = dp.Dataproc(base.ReleaseTrack.GA)
        sparksql_batch = sparksql_batch_factory.SparkSqlBatchFactory(dataproc).UploadLocalFilesAndGetMessage(args)
        return batch_submitter.Submit(sparksql_batch, dataproc, args)