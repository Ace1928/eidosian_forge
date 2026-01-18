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
def _add_self_referential_link(cls, ref, collection_name=None):
    collection_element = collection_name or cls.collection_key
    if cls.api_prefix:
        api_prefix = cls.api_prefix.lstrip('/').rstrip('/')
        api_prefix = _URL_SUBST.sub('{\\1}', api_prefix)
        if flask.request.view_args:
            api_prefix = api_prefix.format(**flask.request.view_args)
        collection_element = '/'.join([api_prefix, collection_name or cls.collection_key])
    self_link = base_url(path='/'.join([collection_element, ref['id']]))
    ref.setdefault('links', {})['self'] = self_link