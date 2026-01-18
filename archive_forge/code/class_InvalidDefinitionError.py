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
class InvalidDefinitionError(Exception):

    def __init__(self, names):
        msg = _('Policies %(names)s are not well defined. Check logs for more details.') % {'names': names}
        super(InvalidDefinitionError, self).__init__(msg)