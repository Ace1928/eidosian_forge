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
def _check_visibility_and_ownership(self, context, image, ownership_check=None):
    if image.visibility != 'shared':
        message = _('Only shared images have members.')
        raise exception.Forbidden(message)
    owner = image.owner
    if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope) and (not context.is_admin):
        if ownership_check == 'create':
            if owner is None or owner != context.owner:
                message = _('You are not permitted to create image members for the image.')
                raise exception.Forbidden(message)
        elif ownership_check == 'update':
            if context.owner == owner:
                message = _("You are not permitted to modify 'status' on this image member.")
                raise exception.Forbidden(message)
        elif ownership_check == 'delete':
            if context.owner != owner:
                message = _('You cannot delete image member.')
                raise exception.Forbidden(message)