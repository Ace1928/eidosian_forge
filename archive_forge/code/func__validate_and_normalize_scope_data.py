from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def _validate_and_normalize_scope_data(self):
    """Validate and normalize scope data."""
    if 'identity' in self.auth:
        if 'application_credential' in self.auth['identity']['methods']:
            if 'scope' in self.auth:
                detail = 'Application credentials cannot request a scope.'
                raise exception.ApplicationCredentialAuthError(detail=detail)
            self._set_scope_from_app_cred(self.auth['identity']['application_credential'])
            return
    if 'scope' not in self.auth:
        return
    if sum(['project' in self.auth['scope'], 'domain' in self.auth['scope'], 'unscoped' in self.auth['scope'], 'system' in self.auth['scope'], 'OS-TRUST:trust' in self.auth['scope']]) != 1:
        msg = 'system, project, domain, OS-TRUST:trust or unscoped'
        raise exception.ValidationError(attribute=msg, target='scope')
    if 'system' in self.auth['scope']:
        self._scope_data = (None, None, None, None, 'all')
        return
    if 'unscoped' in self.auth['scope']:
        self._scope_data = (None, None, None, 'unscoped', None)
        return
    if 'project' in self.auth['scope']:
        project_ref = self._lookup_project(self.auth['scope']['project'])
        self._scope_data = (None, project_ref['id'], None, None, None)
    elif 'domain' in self.auth['scope']:
        domain_ref = self._lookup_domain(self.auth['scope']['domain'])
        self._scope_data = (domain_ref['id'], None, None, None, None)
    elif 'OS-TRUST:trust' in self.auth['scope']:
        trust_ref = self._lookup_trust(self.auth['scope']['OS-TRUST:trust'])
        if trust_ref.get('project_id') is not None:
            project_ref = self._lookup_project({'id': trust_ref['project_id']})
            self._scope_data = (None, project_ref['id'], trust_ref, None, None)
        else:
            self._scope_data = (None, None, trust_ref, None, None)