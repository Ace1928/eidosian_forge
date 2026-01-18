from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _BuildIosDevice(self, device_map):
    return self._messages.IosDevice(iosModelId=device_map['model'], iosVersionId=device_map['version'], locale=device_map['locale'], orientation=device_map['orientation'])