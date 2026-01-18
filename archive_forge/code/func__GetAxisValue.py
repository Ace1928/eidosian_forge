from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetAxisValue(self, dimensionvalue):
    axes = {}
    for dimension in dimensionvalue:
        axes[dimension.key] = dimension.value
    return '{m}-{v}-{l}-{o}'.format(m=axes.get('Model', '?'), v=axes.get('Version', '?'), l=axes.get('Locale', '?'), o=axes.get('Orientation', '?'))