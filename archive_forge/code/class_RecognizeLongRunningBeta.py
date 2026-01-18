from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml.speech import flags
from googlecloudsdk.command_lib.ml.speech import util
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class RecognizeLongRunningBeta(RecognizeLongRunningGA):
    __doc__ = RecognizeLongRunningGA.__doc__
    detailed_help = RecognizeLongRunningGA.detailed_help.copy()
    API_VERSION = 'v1p1beta1'

    @classmethod
    def Args(cls, parser):
        super(RecognizeLongRunningBeta, RecognizeLongRunningBeta).Args(parser)
        cls.flags_mapper.AddBetaRecognizeArgsToParser(parser)

    def MakeRequest(self, args, messages):
        request = super(RecognizeLongRunningBeta, self).MakeRequest(args, messages)
        self.flags_mapper.UpdateBetaArgsInRecognitionConfig(args, request.config)
        return request