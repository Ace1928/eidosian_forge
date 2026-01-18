import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _get_group_ids_for_user_id(self, user_id):
    return [x['id'] for x in PROVIDERS.identity_api.list_groups_for_user(user_id)]