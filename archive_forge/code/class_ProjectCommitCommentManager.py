from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
from .discussions import ProjectCommitDiscussionManager  # noqa: F401
class ProjectCommitCommentManager(ListMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/repository/commits/{commit_id}/comments'
    _obj_cls = ProjectCommitComment
    _from_parent_attrs = {'project_id': 'project_id', 'commit_id': 'id'}
    _create_attrs = RequiredOptional(required=('note',), optional=('path', 'line', 'line_type'))