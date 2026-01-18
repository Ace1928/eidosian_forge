import re
from oslo_config import cfg
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.db import constants as db_constants
def _validate_dns_format(data, max_len=db_constants.FQDN_FIELD_SIZE):
    if not data:
        return
    try:
        trimmed = data[:-1] if data.endswith('.') else data
        if len(trimmed) > max_len:
            raise TypeError(_("'%(trimmed)s' exceeds the %(maxlen)s character FQDN limit") % {'trimmed': trimmed, 'maxlen': max_len})
        labels = trimmed.split('.')
        for label in labels:
            if not label:
                raise TypeError(_('Encountered an empty component'))
            if label.endswith('-') or label.startswith('-'):
                raise TypeError(_("Name '%s' must not start or end with a hyphen") % label)
            if not re.match(constants.DNS_LABEL_REGEX, label):
                raise TypeError(_("Name '%s' must be 1-63 characters long, each of which can only be alphanumeric or a hyphen") % label)
        if len(labels) > 1 and re.match('^[0-9]+$', labels[-1]):
            raise TypeError(_("TLD '%s' must not be all numeric") % labels[-1])
    except TypeError as e:
        msg = _("'%(data)s' not a valid PQDN or FQDN. Reason: %(reason)s") % {'data': data, 'reason': e}
        return msg