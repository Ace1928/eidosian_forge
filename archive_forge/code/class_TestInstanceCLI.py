from novaclient.tests.functional import base
class TestInstanceCLI(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    def test_attach_volume(self):
        """Test we can attach a volume via the cli.

        This test was added after bug 1423695. That bug exposed
        inconsistencies in how to talk to API services from the CLI
        vs. API level. The volumes api calls that were designed to
        populate the completion cache were incorrectly routed to the
        Nova endpoint. Novaclient volumes support actually talks to
        Cinder endpoint directly.

        This would case volume-attach to return a bad error code,
        however it does this *after* the attach command is correctly
        dispatched. So the volume-attach still works, but the user is
        presented a 404 error.

        This test ensures we can do a through path test of: boot,
        create volume, attach volume, detach volume, delete volume,
        destroy.

        """
        name = self.name_generate()
        self.nova('boot', params='--flavor %s --image %s %s --nic net-id=%s --poll' % (self.flavor.name, self.image.name, name, self.network.id))
        servers = self.client.servers.list(search_opts={'name': name})
        self.assertEqual(1, len(servers), servers)
        server = servers[0]
        self.addCleanup(server.delete)
        volume = self.cinder.volumes.create(1)
        self.addCleanup(volume.delete)
        self.wait_for_volume_status(volume, 'available')
        self.nova('volume-attach', params='%s %s' % (name, volume.id))
        self.wait_for_volume_status(volume, 'in-use')
        self.nova('volume-detach', params='%s %s' % (name, volume.id))
        self.wait_for_volume_status(volume, 'available')