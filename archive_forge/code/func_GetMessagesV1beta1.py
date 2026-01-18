from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def GetMessagesV1beta1():
    return apis.GetMessagesModule('containeranalysis', 'v1beta1')