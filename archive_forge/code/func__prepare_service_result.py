from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
@staticmethod
def _prepare_service_result(service):
    http = None
    if service.protocol != 'tcp':
        http = {'cookie_name': to_native(service.http.cookie_name), 'cookie_lifetime': service.http.cookie_name, 'redirect_http': service.http.redirect_http, 'sticky_sessions': service.http.sticky_sessions, 'certificates': [to_native(certificate.name) for certificate in service.http.certificates]}
    health_check = {'protocol': to_native(service.health_check.protocol), 'port': service.health_check.port, 'interval': service.health_check.interval, 'timeout': service.health_check.timeout, 'retries': service.health_check.retries}
    if service.health_check.protocol != 'tcp':
        health_check['http'] = {'domain': to_native(service.health_check.http.domain), 'path': to_native(service.health_check.http.path), 'response': to_native(service.health_check.http.response), 'certificates': [to_native(status_code) for status_code in service.health_check.http.status_codes], 'tls': service.health_check.http.tls}
    return {'protocol': to_native(service.protocol), 'listen_port': service.listen_port, 'destination_port': service.destination_port, 'proxyprotocol': service.proxyprotocol, 'http': http, 'health_check': health_check}