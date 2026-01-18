from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def GetMessageModule():
    return apis.GetMessagesModule('dataplex', 'v1')