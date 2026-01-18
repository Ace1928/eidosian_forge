import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
def _undefined_check(self, check):
    """Check if a RuleCheck references an undefined rule."""
    if isinstance(check, RuleCheck):
        if check.match not in self.rules:
            return True
    rules = getattr(check, 'rules', None)
    if rules:
        for rule in rules:
            if self._undefined_check(rule):
                return True
    return False