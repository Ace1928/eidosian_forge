from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectVariableManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/variables'
    _obj_cls = ProjectVariable
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked', 'environment_scope'))
    _update_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked', 'environment_scope'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectVariable:
        return cast(ProjectVariable, super().get(id=id, lazy=lazy, **kwargs))