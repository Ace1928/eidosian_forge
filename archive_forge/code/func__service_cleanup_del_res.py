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
def _service_cleanup_del_res(self, del_fn, obj, dry_run=True, client_status_queue=None, identified_resources=None, filters=None, resource_evaluation_fn=None):
    need_delete = False
    try:
        if resource_evaluation_fn and callable(resource_evaluation_fn):
            need_del = resource_evaluation_fn(obj, filters, identified_resources)
            if isinstance(need_del, bool):
                need_delete = need_del
        else:
            need_delete = self._service_cleanup_resource_filters_evaluation(obj, filters=filters)
        if need_delete:
            if client_status_queue:
                client_status_queue.put(obj)
            if identified_resources is not None:
                identified_resources[obj.id] = obj
            if not dry_run:
                del_fn(obj)
    except Exception as e:
        self.log.exception('Cannot delete resource %s: %s', obj, str(e))
    return need_delete