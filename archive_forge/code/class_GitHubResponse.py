from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class GitHubResponse(object):

    def __init__(self, response, info):
        self.content = response.read()
        self.info = info

    def json(self):
        return json.loads(self.content)

    def links(self):
        links = {}
        if 'link' in self.info:
            link_header = self.info['link']
            matches = re.findall('<([^>]+)>; rel="([^"]+)"', link_header)
            for url, rel in matches:
                links[rel] = url
        return links