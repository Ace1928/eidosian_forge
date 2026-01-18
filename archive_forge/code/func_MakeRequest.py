from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml.speech import flags
from googlecloudsdk.command_lib.ml.speech import util
def MakeRequest(self, args, messages):
    request = super(RecognizeAlpha, self).MakeRequest(args, messages)
    self.flags_mapper.UpdateAlphaArgsInRecognitionConfig(args, request.config)
    return request