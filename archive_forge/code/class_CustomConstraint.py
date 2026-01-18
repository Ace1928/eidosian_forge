import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
class CustomConstraint(Constraint):
    """A constraint delegating validation to an external class."""
    valid_types = (Schema.STRING_TYPE, Schema.INTEGER_TYPE, Schema.NUMBER_TYPE, Schema.BOOLEAN_TYPE, Schema.LIST_TYPE)

    def __init__(self, name, description=None, environment=None):
        super(CustomConstraint, self).__init__(description)
        self.name = name
        self._environment = environment
        self._custom_constraint = None

    def _constraint(self):
        return self.name

    @property
    def custom_constraint(self):
        if self._custom_constraint is None:
            if self._environment is None:
                self._environment = resources.global_env()
            constraint_class = self._environment.get_constraint(self.name)
            if constraint_class:
                self._custom_constraint = constraint_class()
        return self._custom_constraint

    def _str(self):
        message = getattr(self.custom_constraint, 'message', None)
        if not message:
            message = _('Value must be of type %s') % self.name
        return message

    def _err_msg(self, value):
        constraint = self.custom_constraint
        if constraint is None:
            return _('"%(value)s" does not validate %(name)s (constraint not found)') % {'value': value, 'name': self.name}
        error = getattr(constraint, 'error', None)
        if error:
            return error(value)
        return _('"%(value)s" does not validate %(name)s') % {'value': value, 'name': self.name}

    def _is_valid(self, value, schema, context):
        constraint = self.custom_constraint
        if not constraint:
            return False
        return constraint.validate(value, context)