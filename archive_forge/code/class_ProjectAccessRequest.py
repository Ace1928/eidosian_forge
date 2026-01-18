from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectAccessRequest(AccessRequestMixin, ObjectDeleteMixin, RESTObject):
    pass