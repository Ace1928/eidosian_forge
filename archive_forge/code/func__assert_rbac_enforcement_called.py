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
def _assert_rbac_enforcement_called(resp):
    msg = 'PROGRAMMING ERROR: enforcement (`keystone.common.rbac_enforcer.enforcer.RBACEnforcer.enforce_call()`) has not been called; API is unenforced.'
    g = flask.g
    if flask.request.method != 'OPTIONS':
        assert getattr(g, enforcer._ENFORCEMENT_CHECK_ATTR, False), msg
    return resp