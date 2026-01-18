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
def GetFlexTemplateService():
    return GetClientInstance().projects_locations_flexTemplates