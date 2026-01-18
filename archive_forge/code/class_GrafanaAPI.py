from __future__ import absolute_import, division, print_function
import json
import os
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import basic_auth_header, open_url
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.display import Display
class GrafanaAPI:

    def __init__(self, **kwargs):
        self.grafana_url = kwargs.get('grafana_url', ANSIBLE_GRAFANA_URL)
        self.grafana_api_key = kwargs.get('grafana_api_key', ANSIBLE_GRAFANA_API_KEY)
        self.grafana_user = kwargs.get('grafana_user', ANSIBLE_GRAFANA_USER)
        self.grafana_password = kwargs.get('grafana_password', ANSIBLE_GRAFANA_PASSWORD)
        self.grafana_org_id = kwargs.get('grafana_org_id', ANSIBLE_GRAFANA_ORG_ID)
        self.search = kwargs.get('search', ANSIBLE_GRAFANA_DASHBOARD_SEARCH)

    def grafana_switch_organisation(self, headers):
        try:
            r = open_url('%s/api/user/using/%s' % (self.grafana_url, self.grafana_org_id), headers=headers, method='POST')
        except HTTPError as e:
            raise GrafanaAPIException('Unable to switch to organization %s : %s' % (self.grafana_org_id, to_native(e)))
        if r.getcode() != 200:
            raise GrafanaAPIException('Unable to switch to organization %s : %s' % (self.grafana_org_id, str(r.getcode())))

    def grafana_headers(self):
        headers = {'content-type': 'application/json; charset=utf8'}
        if self.grafana_api_key:
            api_key = self.grafana_api_key
            if len(api_key) % 4 == 2:
                display.deprecated('Passing a mangled version of the API key to the grafana_dashboard lookup is no longer necessary and should not be done.', '2.0.0', collection_name='community.grafana')
                api_key += '=='
            headers['Authorization'] = 'Bearer %s' % api_key
        else:
            headers['Authorization'] = basic_auth_header(self.grafana_user, self.grafana_password)
            self.grafana_switch_organisation(headers)
        return headers

    def grafana_list_dashboards(self):
        headers = self.grafana_headers()
        dashboard_list = []
        try:
            if self.search:
                r = open_url('%s/api/search?query=%s' % (self.grafana_url, self.search), headers=headers, method='GET')
            else:
                r = open_url('%s/api/search/' % self.grafana_url, headers=headers, method='GET')
        except HTTPError as e:
            raise GrafanaAPIException('Unable to search dashboards : %s' % to_native(e))
        if r.getcode() == 200:
            try:
                dashboard_list = json.loads(r.read())
            except Exception as e:
                raise GrafanaAPIException('Unable to parse json list %s' % to_native(e))
        else:
            raise GrafanaAPIException('Unable to list grafana dashboards : %s' % str(r.getcode()))
        return dashboard_list