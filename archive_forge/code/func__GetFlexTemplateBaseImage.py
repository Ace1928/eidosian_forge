from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def _GetFlexTemplateBaseImage(flex_template_base_image):
    """Returns latest base image for given sdk version.

    Args:
        flex_template_base_image: SDK version or base image to use.

    Returns:
      If a custom base image value is given, returns the same value. Else,
      returns the latest base image for the given sdk version.
    """
    if flex_template_base_image == 'JAVA11':
        return Templates.FLEX_TEMPLATE_JAVA11_BASE_IMAGE
    elif flex_template_base_image == 'JAVA17':
        return Templates.FLEX_TEMPLATE_JAVA17_BASE_IMAGE
    elif flex_template_base_image == 'JAVA8':
        return Templates.FLEX_TEMPLATE_JAVA8_BASE_IMAGE
    elif flex_template_base_image == 'PYTHON3':
        return Templates.FLEX_TEMPLATE_PYTHON3_BASE_IMAGE
    elif flex_template_base_image == 'GO':
        return Templates.FLEX_TEMPLATE_GO_BASE_IMAGE
    return flex_template_base_image