from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
import re
import string
import time
from typing import Dict, Optional
from apitools.base.py import exceptions as http_exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
from apitools.base.py import util as http_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.functions import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import http_client
from six.moves import range
def _ValidateUnpackedSourceSize(path: str, ignore_file: Optional[str]=None) -> None:
    """Validate size of unpacked source files."""
    chooser = _GetChooser(path, ignore_file)
    predicate = chooser.IsIncluded
    try:
        size_b = file_utils.GetTreeSizeBytes(path, predicate=predicate)
    except OSError as e:
        raise exceptions.FunctionsError('Error building source archive from path [{path}]. Could not validate source files: [{error}]. Please ensure that path [{path}] contains function code or specify another directory with --source'.format(path=path, error=e))
    size_limit_mb = 512
    size_limit_b = size_limit_mb * 2 ** 20
    if size_b > size_limit_b:
        raise exceptions.OversizedDeploymentError(six.text_type(size_b) + 'B', six.text_type(size_limit_b) + 'B')