import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class ErrorRule(BaseRule):

    def __init__(self, error, **kwargs):
        super().__init__(**kwargs)
        self.error = error

    def evaluate(self, scope_vars, rule_lib):
        """If an error rule's conditions are met, raise an error rule.

        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: EndpointResolutionError
        """
        if self.evaluate_conditions(scope_vars, rule_lib):
            error = rule_lib.resolve_value(self.error, scope_vars)
            raise EndpointResolutionError(msg=error)
        return None