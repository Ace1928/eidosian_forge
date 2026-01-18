import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class EntityNotFound(HeatException):
    msg_fmt = _('The %(entity)s (%(name)s) could not be found.')

    def __init__(self, entity=None, name=None, **kwargs):
        self.entity = entity
        self.name = name
        super(EntityNotFound, self).__init__(entity=entity, name=name, **kwargs)