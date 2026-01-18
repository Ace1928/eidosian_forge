from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
class OpenShiftRoute(AnsibleOpenshiftModule):

    def __init__(self):
        super(OpenShiftRoute, self).__init__(argument_spec=self.argspec, supports_check_mode=True)
        self.append_hash = False
        self.apply = False
        self.warnings = []
        self.params['merge_type'] = None

    @property
    def argspec(self):
        spec = copy.deepcopy(AUTH_ARG_SPEC)
        spec.update(copy.deepcopy(WAIT_ARG_SPEC))
        spec.update(copy.deepcopy(COMMON_ARG_SPEC))
        spec['service'] = dict(type='str', aliases=['svc'])
        spec['namespace'] = dict(required=True, type='str')
        spec['labels'] = dict(type='dict')
        spec['name'] = dict(type='str')
        spec['hostname'] = dict(type='str')
        spec['path'] = dict(type='str')
        spec['wildcard_policy'] = dict(choices=['Subdomain'], type='str')
        spec['port'] = dict(type='str')
        spec['tls'] = dict(type='dict', options=dict(ca_certificate=dict(type='str'), certificate=dict(type='str'), destination_ca_certificate=dict(type='str'), key=dict(type='str', no_log=False), insecure_policy=dict(type='str', choices=['allow', 'redirect', 'disallow'], default='disallow')))
        spec['termination'] = dict(choices=['edge', 'passthrough', 'reencrypt', 'insecure'], default='insecure')
        spec['annotations'] = dict(type='dict')
        return spec

    def execute_module(self):
        service_name = self.params.get('service')
        namespace = self.params['namespace']
        termination_type = self.params.get('termination')
        if termination_type == 'insecure':
            termination_type = None
        state = self.params.get('state')
        if state != 'absent' and (not service_name):
            self.fail_json("If 'state' is not 'absent' then 'service' must be provided")
        custom_wait = self.params.get('wait') and (not self.params.get('wait_condition')) and (state != 'absent')
        if custom_wait:
            self.params['wait'] = False
        route_name = self.params.get('name') or service_name
        labels = self.params.get('labels')
        hostname = self.params.get('hostname')
        path = self.params.get('path')
        wildcard_policy = self.params.get('wildcard_policy')
        port = self.params.get('port')
        annotations = self.params.get('annotations')
        if termination_type and self.params.get('tls'):
            tls_ca_cert = self.params['tls'].get('ca_certificate')
            tls_cert = self.params['tls'].get('certificate')
            tls_dest_ca_cert = self.params['tls'].get('destination_ca_certificate')
            tls_key = self.params['tls'].get('key')
            tls_insecure_policy = self.params['tls'].get('insecure_policy')
            if tls_insecure_policy == 'disallow':
                tls_insecure_policy = None
        else:
            tls_ca_cert = tls_cert = tls_dest_ca_cert = tls_key = tls_insecure_policy = None
        route = {'apiVersion': 'route.openshift.io/v1', 'kind': 'Route', 'metadata': {'name': route_name, 'namespace': namespace, 'labels': labels}, 'spec': {}}
        if annotations:
            route['metadata']['annotations'] = annotations
        if state != 'absent':
            route['spec'] = self.build_route_spec(service_name, namespace, port=port, wildcard_policy=wildcard_policy, hostname=hostname, path=path, termination_type=termination_type, tls_insecure_policy=tls_insecure_policy, tls_ca_cert=tls_ca_cert, tls_cert=tls_cert, tls_key=tls_key, tls_dest_ca_cert=tls_dest_ca_cert)
        result = perform_action(self.svc, route, self.params)
        timeout = self.params.get('wait_timeout')
        sleep = self.params.get('wait_sleep')
        if custom_wait:
            v1_routes = self.find_resource('Route', 'route.openshift.io/v1', fail=True)
            waiter = Waiter(self.client, v1_routes, wait_predicate)
            success, result['result'], result['duration'] = waiter.wait(timeout=timeout, sleep=sleep, name=route_name, namespace=namespace)
        self.exit_json(**result)

    def build_route_spec(self, service_name, namespace, port=None, wildcard_policy=None, hostname=None, path=None, termination_type=None, tls_insecure_policy=None, tls_ca_cert=None, tls_cert=None, tls_key=None, tls_dest_ca_cert=None):
        v1_services = self.find_resource('Service', 'v1', fail=True)
        try:
            target_service = v1_services.get(name=service_name, namespace=namespace)
        except NotFoundError:
            if not port:
                self.fail_json(msg="You need to provide the 'port' argument when exposing a non-existent service")
            target_service = None
        except DynamicApiError as exc:
            self.fail_json(msg='Failed to retrieve service to be exposed: {0}'.format(exc.body), error=exc.status, status=exc.status, reason=exc.reason)
        except Exception as exc:
            self.fail_json(msg='Failed to retrieve service to be exposed: {0}'.format(to_native(exc)), error='', status='', reason='')
        route_spec = {'tls': {}, 'to': {'kind': 'Service', 'name': service_name}, 'port': {'targetPort': self.set_port(target_service, port)}, 'wildcardPolicy': wildcard_policy}
        if termination_type:
            route_spec['tls'] = dict(termination=termination_type.capitalize())
            if tls_insecure_policy:
                if termination_type == 'edge':
                    route_spec['tls']['insecureEdgeTerminationPolicy'] = tls_insecure_policy.capitalize()
                elif termination_type == 'passthrough':
                    if tls_insecure_policy != 'redirect':
                        self.fail_json("'redirect' is the only supported insecureEdgeTerminationPolicy for passthrough routes")
                    route_spec['tls']['insecureEdgeTerminationPolicy'] = tls_insecure_policy.capitalize()
                elif termination_type == 'reencrypt':
                    self.fail_json("'tls.insecure_policy' is not supported with reencrypt routes")
            else:
                route_spec['tls']['insecureEdgeTerminationPolicy'] = None
            if tls_ca_cert:
                if termination_type == 'passthrough':
                    self.fail_json("'tls.ca_certificate' is not supported with passthrough routes")
                route_spec['tls']['caCertificate'] = tls_ca_cert
            if tls_cert:
                if termination_type == 'passthrough':
                    self.fail_json("'tls.certificate' is not supported with passthrough routes")
                route_spec['tls']['certificate'] = tls_cert
            if tls_key:
                if termination_type == 'passthrough':
                    self.fail_json("'tls.key' is not supported with passthrough routes")
                route_spec['tls']['key'] = tls_key
            if tls_dest_ca_cert:
                if termination_type != 'reencrypt':
                    self.fail_json("'destination_certificate' is only valid for reencrypt routes")
                route_spec['tls']['destinationCACertificate'] = tls_dest_ca_cert
        else:
            route_spec['tls'] = None
        if hostname:
            route_spec['host'] = hostname
        if path:
            route_spec['path'] = path
        return route_spec

    def set_port(self, service, port_arg):
        if port_arg:
            return port_arg
        for p in service.spec.ports:
            if p.protocol == 'TCP':
                if p.name is not None:
                    return p.name
                return p.targetPort
        return None