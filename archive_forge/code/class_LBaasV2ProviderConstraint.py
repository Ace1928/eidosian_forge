from heat.engine.clients.os.neutron import neutron_constraints as nc
class LBaasV2ProviderConstraint(nc.ProviderConstraint):
    service_type = 'LOADBALANCERV2'