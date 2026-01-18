from heat.engine.clients.os.neutron import neutron_constraints as nc
class PoolConstraint(nc.NeutronConstraint):
    resource_name = 'pool'
    extension = 'lbaasv2'