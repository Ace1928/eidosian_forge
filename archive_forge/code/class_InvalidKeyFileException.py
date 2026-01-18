from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
class InvalidKeyFileException(Error):
    """There's a problem in a CSEK file."""

    def __init__(self, base_message):
        super(InvalidKeyFileException, self).__init__('{0}\nFor information on proper key file format see: https://cloud.google.com/compute/docs/disks/customer-supplied-encryption#key_file'.format(base_message))