from stevedore import extension
from neutronclient.neutron import v2_0 as neutronV20
class ClientExtensionShow(NeutronClientExtension, neutronV20.ShowCommand):

    def take_action(self, parsed_args):
        return self.execute(parsed_args)

    def execute(self, parsed_args):
        return super(ClientExtensionShow, self).take_action(parsed_args)