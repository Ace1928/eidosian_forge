from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.exceptions import GitlabInvitationError
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import ArrayAttribute, CommaSeparatedListAttribute, RequiredOptional
class InvitationMixin(CRUDMixin):

    def create(self, *args: Any, **kwargs: Any) -> RESTObject:
        invitation = super().create(*args, **kwargs)
        if invitation.status == 'error':
            raise GitlabInvitationError(invitation.message)
        return invitation