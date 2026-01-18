from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.transcoder import util
from googlecloudsdk.command_lib.util.args import labels_util
class ProcessingMode(enum.Enum):
    PROCESSING_MODE_INTERACTIVE = 'PROCESSING_MODE_INTERACTIVE'
    PROCESSING_MODE_BATCH = 'PROCESSING_MODE_BATCH'