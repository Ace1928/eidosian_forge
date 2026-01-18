from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackRouter(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackRouter, self).__init__(module)
        self.returns = {'serviceofferingname': 'service_offering', 'version': 'template_version', 'requiresupgrade': 'requires_upgrade', 'redundantstate': 'redundant_state', 'role': 'role'}
        self.router = None

    def get_service_offering_id(self):
        service_offering = self.module.params.get('service_offering')
        if not service_offering:
            return None
        args = {'issystem': True}
        service_offerings = self.query_api('listServiceOfferings', **args)
        if service_offerings:
            for s in service_offerings['serviceoffering']:
                if service_offering in [s['name'], s['id']]:
                    return s['id']
        self.module.fail_json(msg="Service offering '%s' not found" % service_offering)

    def get_router(self):
        if not self.router:
            router = self.module.params.get('name')
            args = {'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'listall': True, 'fetch_list': True}
            if self.module.params.get('zone'):
                args['zoneid'] = self.get_zone(key='id')
            routers = self.query_api('listRouters', **args)
            if routers:
                for r in routers:
                    if router.lower() in [r['name'].lower(), r['id']]:
                        self.router = r
                        break
        return self.router

    def start_router(self):
        router = self.get_router()
        if not router:
            self.module.fail_json(msg='Router not found')
        if router['state'].lower() != 'running':
            self.result['changed'] = True
            args = {'id': router['id']}
            if not self.module.check_mode:
                res = self.query_api('startRouter', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    router = self.poll_job(res, 'router')
        return router

    def stop_router(self):
        router = self.get_router()
        if not router:
            self.module.fail_json(msg='Router not found')
        if router['state'].lower() != 'stopped':
            self.result['changed'] = True
            args = {'id': router['id']}
            if not self.module.check_mode:
                res = self.query_api('stopRouter', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    router = self.poll_job(res, 'router')
        return router

    def reboot_router(self):
        router = self.get_router()
        if not router:
            self.module.fail_json(msg='Router not found')
        self.result['changed'] = True
        args = {'id': router['id']}
        if not self.module.check_mode:
            res = self.query_api('rebootRouter', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                router = self.poll_job(res, 'router')
        return router

    def absent_router(self):
        router = self.get_router()
        if router:
            self.result['changed'] = True
            args = {'id': router['id']}
            if not self.module.check_mode:
                res = self.query_api('destroyRouter', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'router')
            return router

    def present_router(self):
        router = self.get_router()
        if not router:
            self.module.fail_json(msg='Router can not be created using the API, see cs_network.')
        args = {'id': router['id'], 'serviceofferingid': self.get_service_offering_id()}
        state = self.module.params.get('state')
        if self.has_changed(args, router):
            self.result['changed'] = True
            if not self.module.check_mode:
                current_state = router['state'].lower()
                self.stop_router()
                router = self.query_api('changeServiceForRouter', **args)
                if state in ['restarted', 'started']:
                    router = self.start_router()
                elif state == 'present' and current_state == 'running':
                    router = self.start_router()
        elif state == 'started':
            router = self.start_router()
        elif state == 'stopped':
            router = self.stop_router()
        elif state == 'restarted':
            router = self.reboot_router()
        return router