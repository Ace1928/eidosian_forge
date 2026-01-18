from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class RouterInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(), filters=dict(type='dict', default={}))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        routers = [router.to_dict(computed=False) for router in self.conn.search_routers(name_or_id=self.params['name'], filters=self.params['filters'])]
        self.exit(changed=False, routers=routers)