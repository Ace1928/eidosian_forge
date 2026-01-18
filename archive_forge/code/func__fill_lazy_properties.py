import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def _fill_lazy_properties(self):
    if self._ca_ref and (not self._plugin_name):
        uuid_ref = base.calculate_uuid_ref(self._ca_ref, self._entity)
        result = self._api.get(uuid_ref)
        self._fill_from_data(meta=result.get('meta'), expiration=result.get('expiration'), plugin_name=result.get('plugin_name'), plugin_ca_id=result.get('plugin_ca_id'), created=result.get('created'), updated=result.get('updated'), status=result.get('status'))