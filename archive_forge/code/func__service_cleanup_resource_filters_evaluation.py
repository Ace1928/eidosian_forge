import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _service_cleanup_resource_filters_evaluation(self, obj, filters=None):
    part_cond = []
    if filters is not None and isinstance(filters, dict):
        for k, v in filters.items():
            try:
                res_val = None
                if k == 'created_at' and hasattr(obj, 'created_at'):
                    res_val = getattr(obj, 'created_at')
                if k == 'updated_at' and hasattr(obj, 'updated_at'):
                    res_val = getattr(obj, 'updated_at')
                if res_val:
                    res_date = iso8601.parse_date(res_val)
                    cmp_date = iso8601.parse_date(v)
                    if res_date and cmp_date and (res_date <= cmp_date):
                        part_cond.append(True)
                    else:
                        part_cond.append(False)
                else:
                    self.log.debug('Requested cleanup attribute %s is not available on the resource' % k)
                    part_cond.append(False)
            except Exception:
                self.log.exception('Error during condition evaluation')
    if all(part_cond):
        return True
    else:
        return False