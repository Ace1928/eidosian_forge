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
def AddRules(old_rules, rules_to_add):
    all_threshold_rules = old_rules
    for rule in rules_to_add:
        if rule not in old_rules:
            all_threshold_rules.append(rule)
    return all_threshold_rules