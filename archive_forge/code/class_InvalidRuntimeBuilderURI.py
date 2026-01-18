from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
class InvalidRuntimeBuilderURI(CloudBuildLoadError):
    """Error indicating that the runtime builder URI format wasn't recognized."""

    def __init__(self, uri):
        super(InvalidRuntimeBuilderURI, self).__init__('[{}] is not a valid runtime builder URI. Please set the app/runtime_builders_root property to a URI with either the Google Cloud Storage (`gs://`) or local file (`file://`) protocol.'.format(uri))