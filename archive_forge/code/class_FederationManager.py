from keystoneclient.v3.contrib.federation import domains
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import projects
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import saml
from keystoneclient.v3.contrib.federation import service_providers
class FederationManager(object):

    def __init__(self, api):
        self.identity_providers = identity_providers.IdentityProviderManager(api)
        self.mappings = mappings.MappingManager(api)
        self.protocols = protocols.ProtocolManager(api)
        self.projects = projects.ProjectManager(api)
        self.domains = domains.DomainManager(api)
        self.saml = saml.SamlManager(api)
        self.service_providers = service_providers.ServiceProviderManager(api)