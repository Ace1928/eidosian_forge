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
def _map_context_attributes_into_creds(self, context):
    creds = {}
    context_values = context.to_policy_values()
    for k, v in context_values.items():
        creds[k] = v
    return creds