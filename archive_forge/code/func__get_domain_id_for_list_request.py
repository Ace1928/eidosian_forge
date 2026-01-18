import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@classmethod
def _get_domain_id_for_list_request(cls):
    """Get the domain_id for a v3 list call.

        If we running with multiple domain drivers, then the caller must
        specify a domain_id either as a filter or as part of the token scope.

        """
    if not CONF.identity.domain_specific_drivers_enabled:
        return
    domain_id = flask.request.args.get('domain_id')
    if domain_id:
        return domain_id
    token_ref = cls.get_token_ref()
    if token_ref.domain_scoped:
        return token_ref.domain_id
    elif token_ref.project_scoped:
        return token_ref.project_domain['id']
    elif token_ref.system_scoped:
        return
    else:
        msg = 'No domain information specified as part of list request'
        tr_msg = _('No domain information specified as part of list request')
        LOG.warning(msg)
        raise exception.Unauthorized(tr_msg)