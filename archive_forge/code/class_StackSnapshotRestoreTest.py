from heat_integrationtests.functional import functional_base
class StackSnapshotRestoreTest(functional_base.FunctionalTestsBase):

    def setUp(self):
        super(StackSnapshotRestoreTest, self).setUp()
        if not self.conf.minimal_image_ref:
            raise self.skipException('No image configured to test')
        if not self.conf.minimal_instance_type:
            raise self.skipException('No minimal_instance_type configured to test')
        self.assign_keypair()

    def test_stack_snapshot_restore(self):
        template = '\nheat_template_version: ocata\nparameters:\n  keyname:\n    type: string\n  flavor:\n    type: string\n  image:\n    type: string\n  network:\n    type: string\nresources:\n  my_port:\n    type: OS::Neutron::Port\n    properties:\n      network: {get_param: network}\n  my_server:\n    type: OS::Nova::Server\n    properties:\n      image: {get_param: image}\n      flavor: {get_param: flavor}\n      key_name: {get_param: keyname}\n      networks: [{port: {get_resource: my_port} }]\n\n'

        def get_server_image(server_id):
            server = self.compute_client.servers.get(server_id)
            return server.image['id']
        parameters = {'keyname': self.keypair_name, 'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
        stack_identifier = self.stack_create(template=template, parameters=parameters)
        server_resource = self.client.resources.get(stack_identifier, 'my_server')
        server_id = server_resource.physical_resource_id
        prev_image_id = get_server_image(server_id)
        snapshot_id = self.stack_snapshot(stack_identifier)
        self.stack_restore(stack_identifier, snapshot_id)
        self.assertNotEqual(prev_image_id, get_server_image(server_id))