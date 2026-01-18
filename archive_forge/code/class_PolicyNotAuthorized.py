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
class PolicyNotAuthorized(Exception):
    """Default exception raised for policy enforcement failure."""

    def __init__(self, rule, target, creds):
        msg = _('%(rule)s is disallowed by policy') % {'rule': rule}
        super(PolicyNotAuthorized, self).__init__(msg)