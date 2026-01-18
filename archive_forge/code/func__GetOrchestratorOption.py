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
def _GetOrchestratorOption(self):
    orchestrator_options = self._messages.AndroidInstrumentationTest.OrchestratorOptionValueValuesEnum
    if self._args.use_orchestrator is None:
        return orchestrator_options.ORCHESTRATOR_OPTION_UNSPECIFIED
    elif self._args.use_orchestrator:
        return orchestrator_options.USE_ORCHESTRATOR
    else:
        return orchestrator_options.DO_NOT_USE_ORCHESTRATOR