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
def _CreateImportStager(self, args, compute_holder):
    if _HasExternalCloudProvider(args):
        return ImportFromExternalCloudProviderStager(self.storage_client, compute_holder, args)
    if args.source_image:
        return ImportFromImageStager(self.storage_client, compute_holder, args)
    if daisy_utils.IsLocalFile(args.source_file):
        return ImportFromLocalFileStager(self.storage_client, compute_holder, args)
    try:
        gcs_uri = daisy_utils.MakeGcsObjectUri(args.source_file)
    except storage_util.InvalidObjectNameError:
        raise exceptions.InvalidArgumentException('source-file', 'must be a path to an object in Google Cloud Storage')
    else:
        return ImportFromGSFileStager(self.storage_client, compute_holder, args, gcs_uri)