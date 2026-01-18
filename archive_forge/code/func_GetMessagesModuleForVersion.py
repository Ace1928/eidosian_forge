from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def GetMessagesModuleForVersion(version):
    return apis.GetMessagesModule('looker', version)