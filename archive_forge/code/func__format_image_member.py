import copy
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
def _format_image_member(self, member):
    member_view = {}
    attributes = ['member_id', 'image_id', 'status']
    for key in attributes:
        member_view[key] = getattr(member, key)
    member_view['created_at'] = timeutils.isotime(member.created_at)
    member_view['updated_at'] = timeutils.isotime(member.updated_at)
    member_view['schema'] = '/v2/schemas/member'
    member_view = self.schema.filter(member_view)
    return member_view