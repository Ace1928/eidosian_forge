from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_domain_and_project(self, domain, project):
    if domain and project:
        msg = _('Specify either a domain or project, not both')
        raise exceptions.ValidationError(msg)