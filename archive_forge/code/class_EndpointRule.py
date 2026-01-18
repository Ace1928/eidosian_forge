import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
class EndpointRule(BaseRule):

    def __init__(self, endpoint, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint

    def evaluate(self, scope_vars, rule_lib):
        """Determine if conditions are met to provide a valid endpoint.

        :type scope_vars: dict
        :rtype: RuleSetEndpoint
        """
        if self.evaluate_conditions(scope_vars, rule_lib):
            url = rule_lib.resolve_value(self.endpoint['url'], scope_vars)
            properties = self.resolve_properties(self.endpoint.get('properties', {}), scope_vars, rule_lib)
            headers = self.resolve_headers(scope_vars, rule_lib)
            return RuleSetEndpoint(url=url, properties=properties, headers=headers)
        return None

    def resolve_properties(self, properties, scope_vars, rule_lib):
        """Traverse `properties` attribute, resolving any template strings.

        :type properties: dict/list/str
        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: dict
        """
        if isinstance(properties, list):
            return [self.resolve_properties(prop, scope_vars, rule_lib) for prop in properties]
        elif isinstance(properties, dict):
            return {key: self.resolve_properties(value, scope_vars, rule_lib) for key, value in properties.items()}
        elif rule_lib.is_template(properties):
            return rule_lib.resolve_template_string(properties, scope_vars)
        return properties

    def resolve_headers(self, scope_vars, rule_lib):
        """Iterate through headers attribute resolving all values.

        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: dict
        """
        resolved_headers = {}
        headers = self.endpoint.get('headers', {})
        for header, values in headers.items():
            resolved_headers[header] = [rule_lib.resolve_value(item, scope_vars) for item in values]
        return resolved_headers