from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin
class ProjectIterationManager(ListMixin, RESTManager):
    _path = '/projects/{project_id}/iterations'
    _obj_cls = GroupIteration
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('state', 'search', 'include_ancestors')