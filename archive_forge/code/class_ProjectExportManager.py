from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, DownloadMixin, GetWithoutIdMixin, RefreshMixin
from gitlab.types import RequiredOptional
class ProjectExportManager(GetWithoutIdMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/export'
    _obj_cls = ProjectExport
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(optional=('description',))

    def get(self, **kwargs: Any) -> ProjectExport:
        return cast(ProjectExport, super().get(**kwargs))