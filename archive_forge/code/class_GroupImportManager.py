from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, DownloadMixin, GetWithoutIdMixin, RefreshMixin
from gitlab.types import RequiredOptional
class GroupImportManager(GetWithoutIdMixin, RESTManager):
    _path = '/groups/{group_id}/import'
    _obj_cls = GroupImport
    _from_parent_attrs = {'group_id': 'id'}

    def get(self, **kwargs: Any) -> GroupImport:
        return cast(GroupImport, super().get(**kwargs))