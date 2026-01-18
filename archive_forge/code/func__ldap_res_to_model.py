import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def _ldap_res_to_model(self, res):
    lower_res = {k.lower(): v for k, v in res[1].items()}
    id_attrs = lower_res.get(self.id_attr.lower())
    if not id_attrs:
        message = _('ID attribute %(id_attr)s not found in LDAP object %(dn)s') % {'id_attr': self.id_attr, 'dn': res[0]}
        raise exception.NotFound(message=message)
    if len(id_attrs) > 1:
        message = 'ID attribute %(id_attr)s for LDAP object %(dn)s has multiple values and therefore cannot be used as an ID. Will get the ID from DN instead' % {'id_attr': self.id_attr, 'dn': res[0]}
        LOG.warning(message)
        id_val = self._dn_to_id(res[0])
    else:
        id_val = id_attrs[0]
    obj = self.model(id=id_val)
    for k in obj.known_keys:
        if k in self.attribute_ignore:
            continue
        try:
            map_attr = self.attribute_mapping.get(k, k)
            if map_attr is None:
                continue
            v = lower_res[map_attr.lower()]
        except KeyError:
            pass
        else:
            try:
                value = v[0]
            except IndexError:
                value = None
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    LOG.error('Error decoding value %r (object id %r).', value, res[0])
                    raise
            obj[k] = value
    return obj