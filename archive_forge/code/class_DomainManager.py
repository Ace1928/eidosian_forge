from keystoneclient.v3.contrib.federation import base as federation_base
from keystoneclient.v3 import domains
class DomainManager(federation_base.EntityManager):
    object_type = 'domains'
    resource_class = domains.Domain