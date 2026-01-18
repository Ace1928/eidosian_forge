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
class Modulo(Constraint):
    """Constrain values to modulo.

    Serializes to JSON as::

        {
            'modulo': {'step': <step>, 'offset': <offset>},
            'description': <description>
        }
    """
    STEP, OFFSET = ('step', 'offset')
    valid_types = (Schema.INTEGER_TYPE, Schema.NUMBER_TYPE)

    def __init__(self, step=None, offset=None, description=None):
        super(Modulo, self).__init__(description)
        self.step = step
        self.offset = offset
        if step is None or offset is None:
            raise exception.InvalidSchemaError(message=_('A modulo constraint must have a step value and an offset value specified.'))
        for param in (step, offset):
            if not isinstance(param, (float, int, type(None))):
                raise exception.InvalidSchemaError(message=_('step/offset must be numeric'))
            if not int(param) == param:
                raise exception.InvalidSchemaError(message=_('step/offset must be integer'))
        step, offset = (int(step), int(offset))
        if step == 0:
            raise exception.InvalidSchemaError(message=_('step cannot be 0.'))
        if abs(offset) >= abs(step):
            raise exception.InvalidSchemaError(message=_('offset must be smaller (by absolute value) than step.'))
        if step * offset < 0:
            raise exception.InvalidSchemaError(message=_('step and offset must be both positive or both negative.'))

    def _str(self):
        if self.step is None or self.offset is None:
            fmt = _('The values must be specified.')
        else:
            fmt = _('The value must be a multiple of %(step)s with an offset of %(offset)s.')
        return fmt % self._constraint()

    def _err_msg(self, value):
        return _('%(value)s is not a multiple of %(step)s with an offset of %(offset)s') % {'value': value, 'step': self.step, 'offset': self.offset}

    def _is_valid(self, value, schema, context):
        value = Schema.str_to_num(value)
        if value % self.step != self.offset:
            return False
        return True

    def _constraint(self):

        def constraints():
            if self.step is not None:
                yield (self.STEP, self.step)
            if self.offset is not None:
                yield (self.OFFSET, self.offset)
        return dict(constraints())