from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _BinaryAuthorization(self, args):
    evaluation_mode = flags.GetBinauthzEvaluationMode(args)
    if not evaluation_mode:
        return None
    return self._messages.GoogleCloudGkemulticloudV1BinaryAuthorization(evaluationMode=evaluation_mode)