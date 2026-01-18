from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class VariableManager(CRUDMixin, RESTManager):
    _path = '/admin/ci/variables'
    _obj_cls = Variable
    _create_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked'))
    _update_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> Variable:
        return cast(Variable, super().get(id=id, lazy=lazy, **kwargs))