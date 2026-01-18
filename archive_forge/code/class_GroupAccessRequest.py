from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class GroupAccessRequest(AccessRequestMixin, ObjectDeleteMixin, RESTObject):
    pass