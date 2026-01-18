from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipelineManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/pipelines'
    _obj_cls = ProjectPipeline
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('scope', 'status', 'source', 'ref', 'sha', 'yaml_errors', 'name', 'username', 'order_by', 'sort')
    _create_attrs = RequiredOptional(required=('ref',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectPipeline:
        return cast(ProjectPipeline, super().get(id=id, lazy=lazy, **kwargs))

    def create(self, data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> ProjectPipeline:
        """Creates a new object.

        Args:
            data: Parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request

        Returns:
            A new instance of the managed object class build with
                the data sent by the server
        """
        if TYPE_CHECKING:
            assert self.path is not None
        path = self.path[:-1]
        return cast(ProjectPipeline, CreateMixin.create(self, data, path=path, **kwargs))