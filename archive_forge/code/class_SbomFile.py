from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
class SbomFile(object):
    """Holder for SBOM file's metadata.

  Properties:
    sbom_format: Data format of the SBOM file.
    version: Version of the SBOM format.
    digests: A dictionary of digests, where key is the algorithm.
  """

    def __init__(self, sbom_format, version):
        self._sbom_format = sbom_format
        self._version = version
        self._digests = dict()

    def GetMimeType(self):
        if self._sbom_format == _SBOM_FORMAT_SPDX:
            return _SBOM_REFERENCE_SPDX_MIME_TYPE
        if self._sbom_format == _SBOM_FORMAT_CYCLONEDX:
            return _SBOM_REFERENCE_CYCLONEDX_MIME_TYPE
        return _SBOM_REFERENCE_DEFAULT_MIME_TYPE

    def GetExtension(self):
        if self._sbom_format == _SBOM_FORMAT_SPDX:
            return _SBOM_REFERENCE_SPDX_EXTENSION
        if self._sbom_format == _SBOM_FORMAT_CYCLONEDX:
            return _SBOM_REFERENCE_CYCLONEDX_EXTENSION
        return _SBOM_REFERENCE_DEFAULT_EXTENSION

    @property
    def digests(self):
        return self._digests

    @property
    def sbom_format(self):
        return self._sbom_format

    @property
    def version(self):
        return self._version