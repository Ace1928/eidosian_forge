from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def SqlServerAuditConfig(sql_messages, bucket=None, retention_interval=None, upload_interval=None):
    """Generates the Audit configuration for the instance.

  Args:
    sql_messages: module, The messages module that should be used.
    bucket: string, the GCS bucket name.
    retention_interval: duration, how long to keep generated audit files.
    upload_interval: duration, how often to upload generated audit files.

  Returns:
    sql_messages.SqlServerAuditConfig object.
  """
    if bucket is None and retention_interval is None and (upload_interval is None):
        return None
    config = sql_messages.SqlServerAuditConfig()
    if bucket is not None:
        config.bucket = bucket
    if retention_interval is not None:
        config.retentionInterval = six.text_type(retention_interval) + 's'
    if upload_interval is not None:
        config.uploadInterval = six.text_type(upload_interval) + 's'
    return config