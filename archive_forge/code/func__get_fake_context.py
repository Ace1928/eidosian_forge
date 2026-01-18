import glance_store as store
import webob
import glance.api.v2.image_actions as image_actions
import glance.context
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def _get_fake_context(self, user=USER1, tenant=TENANT1, roles=None, is_admin=False):
    if roles is None:
        roles = ['member']
    kwargs = {'user': user, 'tenant': tenant, 'roles': roles, 'is_admin': is_admin}
    context = glance.context.RequestContext(**kwargs)
    return context