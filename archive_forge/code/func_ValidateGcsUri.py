from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateGcsUri(arg_name):
    """Validates the gcs uri is formatted correctly."""

    def Process(gcs_uri):
        if not gcs_uri.startswith('gs://'):
            raise exceptions.BadArgumentException(arg_name, 'Expected URI {0} to start with `gs://`.'.format(gcs_uri))
        return gcs_uri
    return Process