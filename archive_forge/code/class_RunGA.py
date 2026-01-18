from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import ctrl_c_handler
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.api_lib.firebase.test import history_picker
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import results_bucket
from googlecloudsdk.api_lib.firebase.test import results_summary
from googlecloudsdk.api_lib.firebase.test import tool_results
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.firebase.test.android import arg_manager
from googlecloudsdk.api_lib.firebase.test.android import matrix_creator
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
@base.ReleaseTracks(base.ReleaseTrack.GA)
class RunGA(_BaseRun, base.ListCommand):
    """Invoke a test in Firebase Test Lab for Android and view test results."""

    @staticmethod
    def Args(parser):
        arg_util.AddCommonTestRunArgs(parser)
        arg_util.AddMatrixArgs(parser)
        arg_util.AddAndroidTestArgs(parser)
        arg_util.AddGaArgs(parser)
        base.URI_FLAG.RemoveFromParser(parser)
        parser.display_info.AddFormat(util.OUTCOMES_FORMAT)