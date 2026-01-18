from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ValidateCreditTreatment(unused_ref, args, req):
    """Validates credit treatment matches credit types in filter."""
    budget_tracks_credits = args.IsSpecified('credit_types_treatment') and args.credit_types_treatment == 'include-specified-credits'
    populated_credits_filter = args.IsSpecified('filter_credit_types') and args.filter_credit_types
    if budget_tracks_credits and (not populated_credits_filter):
        raise InvalidBudgetCreditTreatment("'--filter-credit-types' is required when " + "'--credit-types-treatment=include-specified-credits'.")
    if not budget_tracks_credits and populated_credits_filter:
        raise InvalidBudgetCreditTreatment("'--credit-types-treatment' must be 'include-specified-credits' if " + "'--filter-credit-types' is specified.")
    return req