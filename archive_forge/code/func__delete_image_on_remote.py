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
def _delete_image_on_remote(self, image, req):
    """Proxy an image delete to a staging host.

        When an image is staged and then deleted, the staging host still
        has local residue that needs to be cleaned up. If the request to
        delete arrived here, but we are not the stage host, we need to
        proxy it to the appropriate host.

        If the delete succeeds, we return None (per DELETE semantics),
        indicating to the caller that it was handled.

        If the delete fails on the remote end, we allow the
        HTTPClientError to bubble to our caller, which will return the
        error to the client.

        If we fail to contact the remote server, we catch the
        HTTPServerError raised by our proxy method, verify that the
        image still exists, and return it. That indicates to the
        caller that it should proceed with the regular delete logic,
        which will satisfy the client's request, but leave the residue
        on the stage host (which is unavoidable).

        :param image: The Image from the repo
        :param req: The webob.Request for this call
        :returns: None if successful, or a refreshed image if the proxy failed.
        :raises: webob.exc.HTTPClientError if so raised by the remote server.
        """
    try:
        self._proxy_request_to_stage_host(image, req)
    except webob.exc.HTTPServerError:
        return self.gateway.get_repo(req.context).get(image.image_id)