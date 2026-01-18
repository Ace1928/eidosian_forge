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
def _lookup_member(self, req, image, member_id, member_repo=None):
    if not member_repo:
        member_repo = self._get_member_repo(req, image)
    try:
        api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy).get_member()
        return member_repo.get(member_id)
    except exception.NotFound:
        msg = _('%(m_id)s not found in the member list of the image %(i_id)s.') % {'m_id': member_id, 'i_id': image.image_id}
        LOG.warning(msg)
        raise webob.exc.HTTPNotFound(explanation=msg)
    except exception.Forbidden:
        msg = _('You are not authorized to lookup the members of the image %s.') % image.image_id
        LOG.warning(msg)
        raise webob.exc.HTTPForbidden(explanation=msg)