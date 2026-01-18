from oslo_versionedobjects import fields as obj_fields
from neutron_lib._i18n import _
from neutron_lib.services.logapi import constants as log_const
class SecurityEvent(obj_fields.String):

    def __init__(self, valid_values, **kwargs):
        self._valid_values = valid_values
        super().__init__(**kwargs)

    def coerce(self, obj, attr, value):
        if value not in self._valid_values:
            msg = _('Field value %(value)s is not in the list of valid values: %(values)s') % {'value': value, 'values': self._valid_values}
            raise ValueError(msg)
        return super().coerce(obj, attr, value)