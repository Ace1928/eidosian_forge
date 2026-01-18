from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VmwareAboutManager(PyVmomi):

    def __init__(self, module):
        super(VmwareAboutManager, self).__init__(module)

    def gather_about_info(self):
        if not self.content:
            self.module.exit_json(changed=False, about_info=dict())
        about = self.content.about
        self.module.exit_json(changed=False, about_info=dict(product_name=about.name, product_full_name=about.fullName, vendor=about.vendor, version=about.version, build=about.build, locale_version=about.localeVersion, locale_build=about.localeBuild, os_type=about.osType, product_line_id=about.productLineId, api_type=about.apiType, api_version=about.apiVersion, instance_uuid=about.instanceUuid, license_product_name=about.licenseProductName, license_product_version=about.licenseProductVersion))