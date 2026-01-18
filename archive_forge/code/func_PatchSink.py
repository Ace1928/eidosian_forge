from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.trace import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
def PatchSink(self, sink_name, sink_data, update_mask):
    """Patches a sink specified by the arguments."""
    messages = util.GetMessages()
    return util.GetClient().projects_traceSinks.Patch(messages.CloudtraceProjectsTraceSinksPatchRequest(name=sink_name, traceSink=messages.TraceSink(**sink_data), updateMask=','.join(update_mask)))