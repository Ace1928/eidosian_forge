from heat.engine.clients.os.neutron import neutron_constraints as nc
class TapServiceConstraint(nc.NeutronExtConstraint):
    resource_name = 'tap_service'
    extension = 'taas'