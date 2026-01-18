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
def _add_resources(self):
    for r in self.resources:
        c_key = getattr(r, 'collection_key', None)
        m_key = getattr(r, 'member_key', None)
        r_pfx = getattr(r, 'api_prefix', None)
        if not c_key or not m_key:
            LOG.debug('Unable to add resource %(resource)s to API %(name)s, both `member_key` and `collection_key` must be implemented. [collection_key(%(col_key)s) member_key(%(m_key)s)]', {'resource': r.__name__, 'name': self._name, 'col_key': c_key, 'm_key': m_key})
            continue
        if r_pfx != self._api_url_prefix:
            LOG.debug('Unable to add resource %(resource)s to API as the API Prefixes do not match: %(apfx)r != %(rpfx)r', {'resource': r.__name__, 'rpfx': r_pfx, 'apfx': self._api_url_prefix})
            continue
        collection_path = '/%s' % c_key
        if getattr(r, '_id_path_param_name_override', None):
            member_id_key = getattr(r, '_id_path_param_name_override')
        else:
            member_id_key = '%(member_key)s_id' % {'member_key': m_key}
        entity_path = '/%(collection)s/<string:%(member)s>' % {'collection': c_key, 'member': member_id_key}
        jh_e_path = _URL_SUBST.sub('{\\1}', '%(pfx)s/%(e_path)s' % {'pfx': self._api_url_prefix, 'e_path': entity_path.lstrip('/')})
        LOG.debug('Adding standard routes to API %(name)s for `%(resource)s` (API Prefix: %(prefix)s) [%(collection_path)s, %(entity_path)s]', {'name': self._name, 'resource': r.__class__.__name__, 'collection_path': collection_path, 'entity_path': entity_path, 'prefix': self._api_url_prefix})
        self.api.add_resource(r, collection_path, entity_path)
        resource_rel_func = getattr(r, 'json_home_resource_rel_func', json_home.build_v3_resource_relation)
        resource_rel_status = getattr(r, 'json_home_resource_status', None)
        collection_rel_resource_name = getattr(r, 'json_home_collection_resource_name_override', c_key)
        collection_rel = resource_rel_func(resource_name=collection_rel_resource_name)
        href_val = '%(pfx)s%(collection_path)s' % {'pfx': self._api_url_prefix, 'collection_path': collection_path}
        additional_params = getattr(r, 'json_home_additional_parameters', {})
        if additional_params:
            rel_data = dict()
            rel_data['href-template'] = _URL_SUBST.sub('{\\1}', href_val)
            rel_data['href-vars'] = additional_params
        else:
            rel_data = {'href': href_val}
        member_rel_resource_name = getattr(r, 'json_home_member_resource_name_override', m_key)
        entity_rel = resource_rel_func(resource_name=member_rel_resource_name)
        id_str = member_id_key
        parameter_rel_func = getattr(r, 'json_home_parameter_rel_func', json_home.build_v3_parameter_relation)
        id_param_rel = parameter_rel_func(parameter_name=id_str)
        entity_rel_data = {'href-template': jh_e_path, 'href-vars': {id_str: id_param_rel}}
        if additional_params:
            entity_rel_data.setdefault('href-vars', {}).update(additional_params)
        if resource_rel_status is not None:
            json_home.Status.update_resource_data(rel_data, resource_rel_status)
            json_home.Status.update_resource_data(entity_rel_data, resource_rel_status)
        json_home.JsonHomeResources.append_resource(collection_rel, rel_data)
        json_home.JsonHomeResources.append_resource(entity_rel, entity_rel_data)