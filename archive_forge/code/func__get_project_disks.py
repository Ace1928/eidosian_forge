from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _get_project_disks(self, config_data, query):
    """
        project space disk images
        """
    try:
        self._project_disks
    except AttributeError:
        self._project_disks = {}
        request_params = {'maxResults': 500, 'filter': query}
        for project in config_data['projects']:
            session_responses = []
            page_token = True
            while page_token:
                response = self.auth_session.get('https://www.googleapis.com/compute/v1/projects/{0}/aggregated/disks'.format(project), params=request_params)
                response_json = response.json()
                if 'nextPageToken' in response_json:
                    request_params['pageToken'] = response_json['nextPageToken']
                elif 'pageToken' in request_params:
                    del request_params['pageToken']
                if 'items' in response_json:
                    session_responses.append(response_json)
                page_token = 'pageToken' in request_params
        for response in session_responses:
            if 'items' in response:
                for zone_or_region, aggregate in response['items'].items():
                    if 'zones' in zone_or_region:
                        if 'disks' in aggregate:
                            zone = zone_or_region.replace('zones/', '')
                            for disk in aggregate['disks']:
                                if 'zones' in config_data and zone in config_data['zones']:
                                    if 'sourceImage' in disk:
                                        self._project_disks[disk['selfLink']] = disk['sourceImage'].split('/')[-1]
                                    else:
                                        self._project_disks[disk['selfLink']] = disk['selfLink'].split('/')[-1]
                                elif 'sourceImage' in disk:
                                    self._project_disks[disk['selfLink']] = disk['sourceImage'].split('/')[-1]
                                else:
                                    self._project_disks[disk['selfLink']] = disk['selfLink'].split('/')[-1]
    return self._project_disks