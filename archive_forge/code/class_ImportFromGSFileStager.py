from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
class ImportFromGSFileStager(BaseImportFromFileStager):
    """Image import stager from a file in Cloud Storage."""

    def __init__(self, storage_client, compute_holder, args, gcs_uri):
        self.source_file_gcs_uri = gcs_uri
        super(ImportFromGSFileStager, self).__init__(storage_client, compute_holder, args)

    def GetBucketLocation(self):
        return self.storage_client.GetBucketLocationForFile(self.source_file_gcs_uri)

    def _CopySourceFileToScratchBucket(self):
        image_file = os.path.basename(self.source_file_gcs_uri)
        dest_uri = 'gs://{0}/tmpimage/{1}-{2}'.format(self.daisy_bucket, uuid.uuid4(), image_file)
        src_object = resources.REGISTRY.Parse(self.source_file_gcs_uri, collection='storage.objects')
        dest_object = resources.REGISTRY.Parse(dest_uri, collection='storage.objects')
        with progress_tracker.ProgressTracker('Copying [{0}] to [{1}]'.format(self.source_file_gcs_uri, dest_uri)):
            self.storage_client.Rewrite(src_object, dest_object)
        return dest_uri