from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding as api_encoding
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.datastream import camel_case_utils
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
import six
def GetRDBMSV1alpha1ToV1FieldName(field, release_track):
    return RDBMS_FIELD_NAME_BY_RELEASE_TRACK.get(field, {}).get(release_track, field)