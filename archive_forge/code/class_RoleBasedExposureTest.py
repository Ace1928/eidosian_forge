from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
class RoleBasedExposureTest(functional_base.FunctionalTestsBase):
    fl_tmpl = '\nheat_template_version: 2015-10-15\n\nresources:\n  not4everyone:\n    type: OS::Nova::Flavor\n    properties:\n      ram: 20000\n      vcpus: 10\n'
    cvt_tmpl = '\nheat_template_version: 2015-10-15\n\nresources:\n  cvt:\n    type: OS::Cinder::VolumeType\n    properties:\n      name: cvt_test\n'
    host_aggr_tmpl = '\nheat_template_version: 2015-10-15\nparameters:\n  az:\n    type: string\n    default: nova\nresources:\n  cvt:\n    type: OS::Nova::HostAggregate\n    properties:\n      name: aggregate_test\n      availability_zone: {get_param: az}\n'
    scenarios = [('r_nova_flavor', dict(stack_name='s_nova_flavor', template=fl_tmpl, forbidden_r_type='OS::Nova::Flavor', test_creation=True)), ('r_nova_host_aggregate', dict(stack_name='s_nova_ost_aggregate', template=host_aggr_tmpl, forbidden_r_type='OS::Nova::HostAggregate', test_creation=True)), ('r_cinder_vtype', dict(stack_name='s_cinder_vtype', template=cvt_tmpl, forbidden_r_type='OS::Cinder::VolumeType', test_creation=True)), ('r_cinder_vtype_encrypt', dict(forbidden_r_type='OS::Cinder::EncryptedVolumeType', test_creation=False)), ('r_neutron_qos', dict(forbidden_r_type='OS::Neutron::QoSPolicy', test_creation=False)), ('r_neutron_qos_bandwidth_limit', dict(forbidden_r_type='OS::Neutron::QoSBandwidthLimitRule', test_creation=False)), ('r_manila_share_type', dict(forbidden_r_type='OS::Manila::ShareType', test_creation=False))]

    def test_non_admin_forbidden_create_resources(self):
        """Fail to create resource w/o admin role.

        Integration tests job runs as normal OpenStack user,
        and the resources above are configured to require
        admin role in default policy file of Heat.
        """
        if self.test_creation:
            ex = self.assertRaises(exc.Forbidden, self.client.stacks.create, stack_name=self.stack_name, template=self.template)
            self.assertIn(self.forbidden_r_type, ex.message.decode('utf-8'))

    def test_forbidden_resource_not_listed(self):
        resources = self.client.resource_types.list()
        self.assertNotIn(self.forbidden_r_type, (r.resource_type for r in resources))