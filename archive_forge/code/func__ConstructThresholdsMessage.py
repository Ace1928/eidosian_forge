from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def _ConstructThresholdsMessage(thresholds, thresholds_message_class):
    thresholds_message = thresholds_message_class()
    if thresholds is None:
        return None
    thresholds_message.scaleIn = thresholds.scale_in
    thresholds_message.scaleOut = thresholds.scale_out
    return thresholds_message