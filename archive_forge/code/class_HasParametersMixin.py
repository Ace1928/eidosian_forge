import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
class HasParametersMixin:
    """A mixin class for objects that have associated Parameter objects."""

    def params(self, styles, resource=None):
        """Find subsidiary parameters that have the given styles."""
        if resource is None:
            resource = self.resource
        if resource is None:
            raise ValueError('Could not find any particular resource')
        if self.tag is None:
            return []
        param_tags = self.tag.findall(wadl_xpath('param'))
        if param_tags is None:
            return []
        return [Parameter(resource, param_tag) for param_tag in param_tags if param_tag.attrib.get('style') in styles]

    def validate_param_values(self, params, param_values, enforce_completeness=True, **kw_param_values):
        """Make sure the given valueset is valid.

        A valueset might be invalid because it contradicts a fixed
        value or (if enforce_completeness is True) because it lacks a
        required value.

        :param params: A list of Parameter objects.
        :param param_values: A dictionary of parameter values. May include
           paramters whose names are not valid Python identifiers.
        :param enforce_completeness: If True, this method will raise
           an exception when the given value set lacks a value for a
           required parameter.
        :param kw_param_values: A dictionary of parameter values.
        :return: A dictionary of validated parameter values.
        """
        param_values = _merge_dicts(param_values, kw_param_values)
        validated_values = {}
        for param in params:
            name = param.name
            if param.fixed_value is not None:
                if name in param_values and param_values[name] != param.fixed_value:
                    raise ValueError("Value '%s' for parameter '%s' conflicts with fixed value '%s'" % (param_values[name], name, param.fixed_value))
                param_values[name] = param.fixed_value
            options = [option.value for option in param.options]
            if len(options) > 0 and name in param_values and (param_values[name] not in options):
                raise ValueError('Invalid value \'%s\' for parameter \'%s\': valid values are: "%s"' % (param_values[name], name, '", "'.join(options)))
            if enforce_completeness and param.is_required and (not name in param_values):
                raise ValueError("No value for required parameter '%s'" % name)
            if name in param_values:
                validated_values[name] = param_values[name]
                del param_values[name]
        if len(param_values) > 0:
            raise ValueError("Unrecognized parameter(s): '%s'" % "', '".join(param_values.keys()))
        return validated_values