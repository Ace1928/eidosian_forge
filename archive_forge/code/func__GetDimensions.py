from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def _GetDimensions(messages, dimensions):
    if dimensions is None:
        return None
    dt = messages.V1Beta1QuotaOverride.DimensionsValue
    return dt(additionalProperties=[dt.AdditionalProperty(key=k, value=dimensions[k]) for k in sorted(dimensions.keys())])