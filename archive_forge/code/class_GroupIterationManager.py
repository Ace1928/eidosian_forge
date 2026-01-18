from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin
class GroupIterationManager(ListMixin, RESTManager):
    _path = '/groups/{group_id}/iterations'
    _obj_cls = GroupIteration
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('state', 'search', 'include_ancestors')