from typing import Any, Callable, cast, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import FileAttribute, RequiredOptional
class ProjectSecureFileManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/secure_files'
    _obj_cls = ProjectSecureFile
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'file'))
    _types = {'file': FileAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSecureFile:
        return cast(ProjectSecureFile, super().get(id=id, lazy=lazy, **kwargs))