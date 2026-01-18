from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class LoginConfigObject(YamlConfigObject):
    """Auth Login Config file abstraction."""
    PREFERRED_AUTH_KEY = 'spec.preferredAuthentication'
    AUTH_PROVIDERS_KEY = 'spec.authentication'
    CLUSTER_NAME_KEY = 'spec.name'

    @property
    def version(self):
        return self[API_VERSION]

    def _FindMatchingAuthMethod(self, method_name, method_type):
        providers = self.GetAuthProviders(name_only=False)
        found = [x for x in providers if x['name'] == method_name and x[method_type] is not None]
        if found:
            return found.pop()
        return None

    def IsLdap(self):
        """Returns true is the current preferredAuth Method is ldap."""
        try:
            auth_name = self.GetPreferredAuth()
            found_auth = self._FindMatchingAuthMethod(auth_name, 'ldap')
            if found_auth:
                return True
        except (YamlConfigObjectFieldError, KeyError):
            pass
        return False

    def GetPreferredAuth(self):
        if self.version == AUTH_VERSION_2_ALPHA:
            return self[self.PREFERRED_AUTH_KEY]
        else:
            raise YamlConfigObjectFieldError(self.PREFERRED_AUTH_KEY, self.__class__.__name__, 'requires config version [{}]'.format(AUTH_VERSION_2_ALPHA))

    def SetPreferredAuth(self, auth_value):
        if self.version == AUTH_VERSION_2_ALPHA:
            self[self.PREFERRED_AUTH_KEY] = auth_value
        else:
            raise YamlConfigObjectFieldError(self.PREFERRED_AUTH_KEY, self.__class__.__name__, 'requires config version [{}]'.format(AUTH_VERSION_2_ALPHA))

    def GetAuthProviders(self, name_only=True):
        try:
            providers = self[self.AUTH_PROVIDERS_KEY]
        except KeyError:
            return None
        if not providers:
            return None
        if name_only:
            return [provider['name'] for provider in providers]
        return providers