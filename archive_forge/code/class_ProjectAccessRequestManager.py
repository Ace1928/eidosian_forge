from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectAccessRequestManager(ListMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/access_requests'
    _obj_cls = ProjectAccessRequest
    _from_parent_attrs = {'project_id': 'id'}