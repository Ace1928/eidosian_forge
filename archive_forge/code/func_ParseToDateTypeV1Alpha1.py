from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseToDateTypeV1Alpha1(date):
    """Convert the input to Date Type for v1alpha1 Create method."""
    messages = GetMessagesModuleForVersion('v1alpha1')
    return ParseDate(date, messages)