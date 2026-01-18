from openstack import resource
class DomainConfigDriver(resource.Resource):
    driver = resource.Body('driver')