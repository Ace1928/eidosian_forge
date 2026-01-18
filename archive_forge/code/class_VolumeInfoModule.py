from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class VolumeInfoModule(OpenStackModule):
    argument_spec = dict(all_projects=dict(type='bool'), details=dict(type='bool'), name=dict(), status=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['all_projects', 'details', 'name', 'status'] if self.params[k] is not None))
        volumes = [v.to_dict(computed=False) for v in self.conn.block_storage.volumes(**kwargs)]
        self.exit_json(changed=False, volumes=volumes)