import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def _fill_from_data(self, meta=None, expiration=None, plugin_name=None, plugin_ca_id=None, created=None, updated=None, status=None, creator_id=None):
    self._name = None
    self._description = None
    if meta:
        for s in meta:
            key = list(s.keys())[0]
            value = list(s.values())[0]
            if key == 'name':
                self._name = value
            if key == 'description':
                self._description = value
    self._plugin_name = plugin_name
    self._plugin_ca_id = plugin_ca_id
    self._expiration = expiration
    self._creator_id = creator_id
    if self._expiration:
        self._expiration = parse_isotime(self._expiration)
    if self._ca_ref:
        self._status = status
        self._created = created
        self._updated = updated
        if self._created:
            self._created = parse_isotime(self._created)
        if self._updated:
            self._updated = parse_isotime(self._updated)
    else:
        self._status = None
        self._created = None
        self._updated = None