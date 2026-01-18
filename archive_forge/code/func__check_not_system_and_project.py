from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_not_system_and_project(self, system, project):
    if system and project:
        msg = _('Specify either system or project, not both')
        raise exceptions.ValidationError(msg)