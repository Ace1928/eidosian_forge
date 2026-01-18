import datetime
import hashlib
import http.client as http
import os
import re
import urllib.parse as urlparse
import uuid
from castellan.common import exception as castellan_exception
from castellan import key_manager
import glance_store
from glance_store import location
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from oslo_utils import encodeutils
from oslo_utils import timeutils as oslo_timeutils
import requests
import webob.exc
from glance.api import common
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import store_utils
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
from glance import context as glance_context
import glance.db
import glance.gateway
from glance.i18n import _, _LE, _LI, _LW
import glance.notifier
from glance.quota import keystone as ks_quota
import glance.schema
def _do_replace_locations(self, image, value):
    if CONF.show_multiple_locations == False:
        msg = _("It's not allowed to update locations if locations are invisible.")
        raise webob.exc.HTTPForbidden(explanation=msg)
    if image.status not in ('active', 'queued'):
        msg = _("It's not allowed to replace locations if image status is %s.") % image.status
        raise webob.exc.HTTPConflict(explanation=msg)
    val_data = self._validate_validation_data(image, value)
    updated_location = value
    if CONF.enabled_backends:
        updated_location = store_utils.get_updated_store_location(value)
    try:
        image.locations = updated_location
        if image.status == 'queued':
            for k, v in val_data.items():
                setattr(image, k, v)
            image.status = 'active'
    except (exception.BadStoreUri, exception.DuplicateLocation) as e:
        raise webob.exc.HTTPBadRequest(explanation=e.msg)
    except ValueError as ve:
        raise webob.exc.HTTPBadRequest(explanation=encodeutils.exception_to_unicode(ve))