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
def _add_mapped_resources(self):
    for r in self.resource_mapping:
        alt_url_json_home_data = []
        LOG.debug('Adding resource routes to API %(name)s: [%(url)r %(kwargs)r]', {'name': self._name, 'url': r.url, 'kwargs': r.kwargs})
        urls = [r.url]
        if r.alternate_urls is not None:
            for element in r.alternate_urls:
                if self._api_url_prefix:
                    LOG.debug('Unable to add additional resource route `%(route)s` to API %(name)s because API has a URL prefix. Only APIs without explicit prefixes can have alternate URL routes added.', {'route': element['url'], 'name': self._name})
                    continue
                LOG.debug('Adding additional resource route (alternate) to API %(name)s: [%(url)r %(kwargs)r]', {'name': self._name, 'url': element['url'], 'kwargs': r.kwargs})
                urls.append(element['url'])
                if element.get('json_home'):
                    alt_url_json_home_data.append(element['json_home'])
        self.api.add_resource(r.resource, *urls, **r.kwargs)
        if r.json_home_data:
            resource_data = {}
            conv_url = '%(pfx)s/%(url)s' % {'url': _URL_SUBST.sub('{\\1}', r.url).lstrip('/'), 'pfx': self._api_url_prefix}
            if r.json_home_data.path_vars:
                resource_data['href-template'] = conv_url
                resource_data['href-vars'] = r.json_home_data.path_vars
            else:
                resource_data['href'] = conv_url
            json_home.Status.update_resource_data(resource_data, r.json_home_data.status)
            json_home.JsonHomeResources.append_resource(r.json_home_data.rel, resource_data)
            for element in alt_url_json_home_data:
                json_home.JsonHomeResources.append_resource(element.rel, resource_data)