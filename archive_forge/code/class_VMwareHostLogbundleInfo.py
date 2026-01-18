from __future__ import absolute_import, division, print_function
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
import xml.etree.ElementTree as ET
class VMwareHostLogbundleInfo(PyVmomi):

    def __init__(self, module):
        super(VMwareHostLogbundleInfo, self).__init__(module)
        self.esxi_hostname = self.params['esxi_hostname']

    def generate_req_headers(self, url):
        req = vim.SessionManager.HttpServiceRequestSpec(method='httpGet', url=url)
        ticket = self.content.sessionManager.AcquireGenericServiceTicket(req)
        headers = {'Content-Type': 'application/octet-stream', 'Cookie': 'vmware_cgi_ticket=%s' % ticket.id}
        return headers

    def get_listmanifests(self):
        url = 'https://' + self.esxi_hostname + '/cgi-bin/vm-support.cgi?listmanifests=1'
        headers = self.generate_req_headers(url)
        try:
            resp, info = fetch_url(self.module, method='GET', headers=headers, url=url)
            manifest_list = ET.fromstring(resp.read())
            manifests = []
            for manifest in manifest_list[0]:
                manifests.append(manifest.attrib)
            self.module.exit_json(changed=False, manifests=manifests)
        except Exception as e:
            self.module.fail_json(msg='Failed to fetch manifests from %s: %s' % (url, e))