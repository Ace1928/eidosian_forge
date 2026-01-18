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
def _BuildIosTestLoopTestSpec(self):
    """Build a TestSpecification for an IosXcTest."""
    spec = self._messages.TestSpecification(disableVideoRecording=not self._args.record_video, iosTestSetup=self._BuildGenericTestSetup(), testTimeout=matrix_ops.ReformatDuration(self._args.timeout), iosTestLoop=self._messages.IosTestLoop(appIpa=self._BuildFileReference(self._args.app), scenarios=self._args.scenario_numbers))
    return spec