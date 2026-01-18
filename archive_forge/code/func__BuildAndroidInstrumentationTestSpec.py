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
import six
def _BuildAndroidInstrumentationTestSpec(self):
    """Build a TestSpecification for an AndroidInstrumentationTest."""
    spec = self._BuildGenericTestSpec()
    app_apk, app_bundle = self._BuildAppReference(self._args.app)
    spec.androidInstrumentationTest = self._messages.AndroidInstrumentationTest(appApk=app_apk, appBundle=app_bundle, testApk=self._BuildFileReference(os.path.basename(self._args.test)), appPackageId=self._args.app_package, testPackageId=self._args.test_package, testRunnerClass=self._args.test_runner_class, testTargets=self._args.test_targets or [], orchestratorOption=self._GetOrchestratorOption(), shardingOption=self._BuildShardingOption())
    return spec