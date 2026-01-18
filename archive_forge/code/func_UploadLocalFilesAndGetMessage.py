from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
def UploadLocalFilesAndGetMessage(self, args):
    """Uploads local files and creates a SparkSqlBatch message.

    Uploads user local files and change the URIs to local files to uploaded
    URIs.
    Creates a SparkSqlBatch message.

    Args:
      args: Parsed arguments.

    Returns:
      A SparkSqlBatch message instance.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    """
    kwargs = {}
    dependencies = {}
    dependencies['queryFileUri'] = [args.SQL_SCRIPT]
    if args.jars:
        dependencies['jarFileUris'] = args.jars
    params = args.vars
    if params:
        kwargs['queryVariables'] = encoding.DictToAdditionalPropertyMessage(params, self.dataproc.messages.SparkSqlBatch.QueryVariablesValue, sort_items=True)
    if local_file_uploader.HasLocalFiles(dependencies):
        if not args.deps_bucket:
            raise AttributeError('--deps-bucket was not specified.')
        dependencies = local_file_uploader.Upload(args.deps_bucket, dependencies)
    dependencies['queryFileUri'] = dependencies['queryFileUri'][0]
    kwargs.update(dependencies)
    return self.dataproc.messages.SparkSqlBatch(**kwargs)