from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import BadgeRenderMixin, CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectBadgeManager(BadgeRenderMixin, CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/badges'
    _obj_cls = ProjectBadge
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('link_url', 'image_url'))
    _update_attrs = RequiredOptional(optional=('link_url', 'image_url'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectBadge:
        return cast(ProjectBadge, super().get(id=id, lazy=lazy, **kwargs))