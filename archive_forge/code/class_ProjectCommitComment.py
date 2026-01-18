from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
from .discussions import ProjectCommitDiscussionManager  # noqa: F401
class ProjectCommitComment(RESTObject):
    _id_attr = None
    _repr_attr = 'note'