from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
class Scaleway(object):

    def __init__(self, module):
        self.module = module
        self.headers = {'X-Auth-Token': self.module.params.get('api_token'), 'User-Agent': self.get_user_agent_string(module), 'Content-Type': 'application/json'}
        self.name = None

    def get_resources(self):
        results = self.get('/%s' % self.name)
        if not results.ok:
            raise ScalewayException('Error fetching {0} ({1}) [{2}: {3}]'.format(self.name, '%s/%s' % (self.module.params.get('api_url'), self.name), results.status_code, results.json['message']))
        return results.json.get(self.name)

    def _url_builder(self, path, params):
        d = self.module.params.get('query_parameters')
        if params is not None:
            d.update(params)
        query_string = urlencode(d, doseq=True)
        if path[0] == '/':
            path = path[1:]
        return '%s/%s?%s' % (self.module.params.get('api_url'), path, query_string)

    def send(self, method, path, data=None, headers=None, params=None):
        url = self._url_builder(path=path, params=params)
        self.warn(url)
        if headers is not None:
            self.headers.update(headers)
        if self.headers['Content-Type'] == 'application/json':
            data = self.module.jsonify(data)
        resp, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method, timeout=self.module.params.get('api_timeout'))
        if info['status'] == -1:
            self.module.fail_json(msg=info['msg'])
        return Response(resp, info)

    @staticmethod
    def get_user_agent_string(module):
        return 'ansible %s Python %s' % (module.ansible_version, sys.version.split(' ', 1)[0])

    def get(self, path, data=None, headers=None, params=None):
        return self.send(method='GET', path=path, data=data, headers=headers, params=params)

    def put(self, path, data=None, headers=None, params=None):
        return self.send(method='PUT', path=path, data=data, headers=headers, params=params)

    def post(self, path, data=None, headers=None, params=None):
        return self.send(method='POST', path=path, data=data, headers=headers, params=params)

    def delete(self, path, data=None, headers=None, params=None):
        return self.send(method='DELETE', path=path, data=data, headers=headers, params=params)

    def patch(self, path, data=None, headers=None, params=None):
        return self.send(method='PATCH', path=path, data=data, headers=headers, params=params)

    def update(self, path, data=None, headers=None, params=None):
        return self.send(method='UPDATE', path=path, data=data, headers=headers, params=params)

    def warn(self, x):
        self.module.warn(str(x))

    def fetch_state(self, resource):
        self.module.debug('fetch_state of resource: %s' % resource['id'])
        response = self.get(path=self.api_path + '/%s' % resource['id'])
        if response.status_code == 404:
            return 'absent'
        if not response.ok:
            msg = 'Error during state fetching: (%s) %s' % (response.status_code, response.json)
            self.module.fail_json(msg=msg)
        try:
            self.module.debug('Resource %s in state: %s' % (resource['id'], response.json['status']))
            return response.json['status']
        except KeyError:
            self.module.fail_json(msg='Could not fetch state in %s' % response.json)

    def fetch_paginated_resources(self, resource_key, **pagination_kwargs):
        response = self.get(path=self.api_path, params=pagination_kwargs)
        status_code = response.status_code
        if not response.ok:
            self.module.fail_json(msg='Error getting {0} [{1}: {2}]'.format(resource_key, response.status_code, response.json['message']))
        return response.json[resource_key]

    def fetch_all_resources(self, resource_key, **pagination_kwargs):
        resources = []
        result = [None]
        while len(result) != 0:
            result = self.fetch_paginated_resources(resource_key, **pagination_kwargs)
            resources += result
            if 'page' in pagination_kwargs:
                pagination_kwargs['page'] += 1
            else:
                pagination_kwargs['page'] = 2
        return resources

    def wait_to_complete_state_transition(self, resource, stable_states, force_wait=False):
        wait = self.module.params['wait']
        if not (wait or force_wait):
            return
        wait_timeout = self.module.params['wait_timeout']
        wait_sleep_time = self.module.params['wait_sleep_time']
        time.sleep(wait_sleep_time)
        start = datetime.datetime.utcnow()
        end = start + datetime.timedelta(seconds=wait_timeout)
        while datetime.datetime.utcnow() < end:
            self.module.debug('We are going to wait for the resource to finish its transition')
            state = self.fetch_state(resource)
            if state in stable_states:
                self.module.debug('It seems that the resource is not in transition anymore.')
                self.module.debug('load-balancer in state: %s' % self.fetch_state(resource))
                break
            time.sleep(wait_sleep_time)
        else:
            self.module.fail_json(msg='Server takes too long to finish its transition')