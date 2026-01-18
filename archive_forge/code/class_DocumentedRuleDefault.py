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
class DocumentedRuleDefault(RuleDefault):
    """A class for holding policy-in-code policy objects definitions

    This class provides the same functionality as the RuleDefault class, but it
    also requires additional data about the policy rule being registered. This
    is necessary so that proper documentation can be rendered based on the
    attributes of this class. Eventually, all usage of RuleDefault should be
    converted to use DocumentedRuleDefault.

    :param operations: List of dicts containing each API URL and
        corresponding http request method.

        Example::

            operations=[{'path': '/foo', 'method': 'GET'},
                        {'path': '/some', 'method': 'POST'}]
    """

    def __init__(self, name, check_str, description, operations, deprecated_rule=None, deprecated_for_removal=False, deprecated_reason=None, deprecated_since=None, scope_types=None):
        super().__init__(name, check_str, description, deprecated_rule=deprecated_rule, deprecated_for_removal=deprecated_for_removal, deprecated_reason=deprecated_reason, deprecated_since=deprecated_since, scope_types=scope_types)
        self._operations = operations
        if not self._description:
            raise InvalidRuleDefault('Description is required')
        if not isinstance(self._operations, list):
            raise InvalidRuleDefault('Operations must be a list')
        if not self._operations:
            raise InvalidRuleDefault('Operations list must not be empty')
        for op in self._operations:
            if 'path' not in op:
                raise InvalidRuleDefault('Operation must contain a path')
            if 'method' not in op:
                raise InvalidRuleDefault('Operation must contain a method')
            if len(op.keys()) > 2:
                raise InvalidRuleDefault('Operation contains > 2 keys')

    @property
    def description(self):
        return self._description

    @property
    def operations(self):
        return self._operations