from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from six.moves.urllib import parse
import uritemplate
def _GenericErrorMessage(matrix):
    return '\nMatrix [{m}] unexpectedly reached final status {s} without returning a URL to any test results in the Firebase console. Please re-check the validity of your test files and parameters and try again.'.format(m=matrix.testMatrixId, s=matrix.state)