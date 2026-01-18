from openstack.cloud import _utils
from openstack import exceptions
from openstack.orchestration.util import event_utils
from openstack.orchestration.v1._proxy import Proxy
def _search_one_stack(name_or_id=None, filters=None):
    try:
        stack = self.orchestration.find_stack(name_or_id, ignore_missing=False, resolve_outputs=resolve_outputs)
        if stack.status == 'DELETE_COMPLETE':
            return []
    except exceptions.NotFoundException:
        return []
    return _utils._filter_list([stack], name_or_id, filters)