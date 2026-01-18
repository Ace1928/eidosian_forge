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
def _ExtractInvalidMatrixDetails(matrix):
    invalid_details_for_user = []
    for invalid_detail in matrix.extendedInvalidMatrixDetails:
        invalid_details_for_user.append(f'Reason: {invalid_detail.reason} Message: {invalid_detail.message}')
    if invalid_details_for_user:
        return 'Matrix [{m}] failed during validation.\n{msg}'.format(m=matrix.testMatrixId, msg=os.linesep.join(invalid_details_for_user))
    else:
        return _GetLegacyInvalidMatrixDetails(matrix)