from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_transfer_spec(job, args, messages):
    """Creates or modifies TransferSpec based on args."""
    if not job.transferSpec:
        job.transferSpec = messages.TransferSpec()
    if getattr(args, 'source', None):
        job.transferSpec.httpDataSource = None
        job.transferSpec.posixDataSource = None
        job.transferSpec.gcsDataSource = None
        job.transferSpec.awsS3CompatibleDataSource = None
        job.transferSpec.awsS3DataSource = None
        job.transferSpec.azureBlobStorageDataSource = None
        job.transferSpec.hdfsDataSource = None
        try:
            source_url = storage_url.storage_url_from_string(args.source)
        except errors.InvalidUrlError:
            if args.source.startswith(storage_url.ProviderPrefix.HTTP.value):
                job.transferSpec.httpDataSource = messages.HttpData(listUrl=args.source)
                source_url = None
            else:
                raise
        else:
            if source_url.scheme is storage_url.ProviderPrefix.FILE:
                source_url = _prompt_and_add_valid_scheme(source_url, VALID_SOURCE_TRANSFER_SCHEMES)
            if source_url.scheme is storage_url.ProviderPrefix.HDFS:
                job.transferSpec.hdfsDataSource = messages.HdfsData(path=source_url.object_name)
            if source_url.scheme is storage_url.ProviderPrefix.POSIX:
                job.transferSpec.posixDataSource = messages.PosixFilesystem(rootDirectory=source_url.object_name)
            elif source_url.scheme is storage_url.ProviderPrefix.GCS:
                job.transferSpec.gcsDataSource = messages.GcsData(bucketName=source_url.bucket_name, path=source_url.object_name)
            elif source_url.scheme is storage_url.ProviderPrefix.S3:
                if args.source_endpoint:
                    job.transferSpec.awsS3CompatibleDataSource = messages.AwsS3CompatibleData(bucketName=source_url.bucket_name, endpoint=args.source_endpoint, path=source_url.object_name, region=args.source_signing_region, s3Metadata=_get_s3_compatible_metadata(args, messages))
                else:
                    job.transferSpec.awsS3DataSource = messages.AwsS3Data(bucketName=source_url.bucket_name, path=source_url.object_name)
            elif isinstance(source_url, storage_url.AzureUrl):
                job.transferSpec.azureBlobStorageDataSource = messages.AzureBlobStorageData(container=source_url.bucket_name, path=source_url.object_name, storageAccount=source_url.account)
    if getattr(args, 'destination', None):
        job.transferSpec.posixDataSink = None
        job.transferSpec.gcsDataSink = None
        destination_url = storage_url.storage_url_from_string(args.destination)
        if destination_url.scheme is storage_url.ProviderPrefix.FILE:
            destination_url = _prompt_and_add_valid_scheme(destination_url, VALID_DESTINATION_TRANSFER_SCHEMES)
        if destination_url.scheme is storage_url.ProviderPrefix.GCS:
            job.transferSpec.gcsDataSink = messages.GcsData(bucketName=destination_url.bucket_name, path=destination_url.object_name)
        elif destination_url.scheme is storage_url.ProviderPrefix.POSIX:
            job.transferSpec.posixDataSink = messages.PosixFilesystem(rootDirectory=destination_url.object_name)
    if getattr(args, 'destination_agent_pool', None):
        job.transferSpec.sinkAgentPoolName = name_util.add_agent_pool_prefix(args.destination_agent_pool)
    if getattr(args, 'source_agent_pool', None):
        job.transferSpec.sourceAgentPoolName = name_util.add_agent_pool_prefix(args.source_agent_pool)
    if getattr(args, 'intermediate_storage_path', None):
        intermediate_storage_url = storage_url.storage_url_from_string(args.intermediate_storage_path)
        job.transferSpec.gcsIntermediateDataLocation = messages.GcsData(bucketName=intermediate_storage_url.bucket_name, path=intermediate_storage_url.object_name)
    if getattr(args, 'manifest_file', None):
        job.transferSpec.transferManifest = messages.TransferManifest(location=args.manifest_file)
    _create_or_modify_creds(job.transferSpec, args, messages)
    _create_or_modify_object_conditions(job.transferSpec, args, messages)
    _create_or_modify_transfer_options(job.transferSpec, args, messages)