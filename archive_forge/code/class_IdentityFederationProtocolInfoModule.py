from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityFederationProtocolInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(aliases=['id']), idp=dict(required=True, aliases=['idp_id', 'idp_name']))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        id = self.params['name']
        idp_id = self.params['idp']
        if id:
            protocol = self.conn.identity.find_federation_protocol(idp_id, id)
            protocols = [protocol] if protocol else []
        else:
            protocols = self.conn.identity.federation_protocols(idp_id)
        self.exit_json(changed=False, protocols=[p.to_dict(computed=False) for p in protocols])