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
def CheckMoneyRegex(input_string):
    accepted_regex = re.compile('^[0-9]*.?[0-9]+([a-zA-Z]{3})?$')
    if not re.match(accepted_regex, input_string):
        raise InvalidBudgetAmountInput('The input is not valid for --budget-amount. It must be an int or float with an optional 3-letter currency code.')