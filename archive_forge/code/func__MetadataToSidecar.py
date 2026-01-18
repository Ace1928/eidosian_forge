from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def _MetadataToSidecar(metadata):
    if 'annotations' in metadata and _MANAGED_SIDECARS_ANNOTATION in metadata['annotations']:
        return metadata['annotations'][_MANAGED_SIDECARS_ANNOTATION]
    return None