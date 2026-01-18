from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _enforce_mutually_exclusive_group(self, system, domain, project):
    if not system:
        if domain and project:
            msg = _('Specify either a domain or project, not both')
            raise exceptions.ValidationError(msg)
        elif not (domain or project):
            msg = _('Must specify either system, domain, or project')
            raise exceptions.ValidationError(msg)
    elif system:
        if domain and project:
            msg = _('Specify either system, domain, or project, not all three.')
            raise exceptions.ValidationError(msg)
        if domain:
            msg = _('Specify either system or a domain, not both')
            raise exceptions.ValidationError(msg)
        if project:
            msg = _('Specify either a system or project, not both')
            raise exceptions.ValidationError(msg)