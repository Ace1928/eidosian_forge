from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
class Balancer(object):
    """ Apache httpd 2.4 mod_proxy balancer object"""

    def __init__(self, host, suffix, module, members=None, tls=False):
        if tls:
            self.base_url = 'https://' + str(host)
            self.url = 'https://' + str(host) + str(suffix)
        else:
            self.base_url = 'http://' + str(host)
            self.url = 'http://' + str(host) + str(suffix)
        self.module = module
        self.page = self.fetch_balancer_page()
        if members is None:
            self._members = []

    def fetch_balancer_page(self):
        """ Returns the balancer management html page as a string for later parsing."""
        page = fetch_url(self.module, str(self.url))
        if page[1]['status'] != 200:
            self.module.fail_json(msg='Could not get balancer page! HTTP status response: ' + str(page[1]['status']))
        else:
            content = page[0].read()
            apache_version = regexp_extraction(content.upper(), APACHE_VERSION_EXPRESSION, 1)
            if apache_version:
                if not re.search(pattern='2\\.4\\.[\\d]*', string=apache_version):
                    self.module.fail_json(msg='This module only acts on an Apache2 2.4+ instance, current Apache2 version: ' + str(apache_version))
                return content
            else:
                self.module.fail_json(msg='Could not get the Apache server version from the balancer-manager')

    def get_balancer_members(self):
        """ Returns members of the balancer as a generator object for later iteration."""
        try:
            soup = BeautifulSoup(self.page)
        except TypeError:
            self.module.fail_json(msg='Cannot parse balancer page HTML! ' + str(self.page))
        else:
            for element in soup.findAll('a')[1::1]:
                balancer_member_suffix = str(element.get('href'))
                if not balancer_member_suffix:
                    self.module.fail_json(msg="Argument 'balancer_member_suffix' is empty!")
                else:
                    yield BalancerMember(str(self.base_url + balancer_member_suffix), str(self.url), self.module)
    members = property(get_balancer_members)