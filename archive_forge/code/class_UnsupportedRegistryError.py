from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.docker import client_lib
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
import six
class UnsupportedRegistryError(client_lib.DockerError):
    """Indicates an attempt to use an unsupported registry."""

    def __init__(self, image_url):
        self.image_url = image_url

    def __str__(self):
        return '{0} is not in a supported registry.  Supported registries are {1}'.format(self.image_url, constants.ALL_SUPPORTED_REGISTRIES)