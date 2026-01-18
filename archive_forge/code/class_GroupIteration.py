from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin
class GroupIteration(RESTObject):
    _repr_attr = 'title'