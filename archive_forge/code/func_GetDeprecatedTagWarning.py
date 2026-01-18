from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetDeprecatedTagWarning(models, platform='android'):
    """Returns a warning string iff any device model is marked deprecated."""
    for model in models:
        for tag in model.tags:
            if 'deprecated' in tag:
                return 'Some devices are deprecated. Learn more at https://firebase.google.com/docs/test-lab/%s/available-testing-devices#deprecated' % platform
    return None