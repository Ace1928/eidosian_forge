from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def __get_service_http(self, http_arg):
    service_http = LoadBalancerServiceHttp(certificates=[])
    if http_arg.get('cookie_name') is not None:
        service_http.cookie_name = http_arg.get('cookie_name')
    if http_arg.get('cookie_lifetime') is not None:
        service_http.cookie_lifetime = http_arg.get('cookie_lifetime')
    if http_arg.get('sticky_sessions') is not None:
        service_http.sticky_sessions = http_arg.get('sticky_sessions')
    if http_arg.get('redirect_http') is not None:
        service_http.redirect_http = http_arg.get('redirect_http')
    if http_arg.get('certificates') is not None:
        certificates = http_arg.get('certificates')
        if certificates is not None:
            for certificate in certificates:
                hcloud_cert = None
                try:
                    try:
                        hcloud_cert = self.client.certificates.get_by_name(certificate)
                    except Exception:
                        hcloud_cert = self.client.certificates.get_by_id(certificate)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
                service_http.certificates.append(hcloud_cert)
    return service_http