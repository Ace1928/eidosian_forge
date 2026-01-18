from typing import Any, cast, Dict, Optional, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectCluster(SaveMixin, ObjectDeleteMixin, RESTObject):
    pass