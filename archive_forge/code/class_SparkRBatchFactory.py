from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
class SparkRBatchFactory(object):
    """Factory class for SparkRBatch message."""

    def __init__(self, dataproc):
        """Factory class for SparkRBatch message.

    Args:
      dataproc: A Dataproc instance.
    """
        self.dataproc = dataproc

    def UploadLocalFilesAndGetMessage(self, args):
        """Upload local files and creates a SparkRBatch message.

    Upload user local files and change local file URIs to point to the uploaded
    URIs.
    Creates a SparkRBatch message based on parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      A SparkRBatch message.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    """
        kwargs = {}
        if args.args:
            kwargs['args'] = args.args
        dependencies = {}
        dependencies['mainRFileUri'] = [args.MAIN_R_FILE]
        if args.files:
            dependencies['fileUris'] = args.files
        if args.archives:
            dependencies['archiveUris'] = args.archives
        if local_file_uploader.HasLocalFiles(dependencies):
            if not args.deps_bucket:
                raise AttributeError('--deps-bucket was not specified.')
            dependencies = local_file_uploader.Upload(args.deps_bucket, dependencies)
        dependencies['mainRFileUri'] = dependencies['mainRFileUri'][0]
        kwargs.update(dependencies)
        return self.dataproc.messages.SparkRBatch(**kwargs)