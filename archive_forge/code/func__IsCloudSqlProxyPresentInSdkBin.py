from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import subprocess
import time
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def _IsCloudSqlProxyPresentInSdkBin(cloud_sql_proxy_path):
    """Checks if cloud_sql_proxy_path binary is present in cloud sdk bin."""
    return os.path.exists(cloud_sql_proxy_path) and os.access(cloud_sql_proxy_path, os.X_OK)