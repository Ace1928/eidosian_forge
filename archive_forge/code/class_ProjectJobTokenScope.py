from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectJobTokenScope(RefreshMixin, SaveMixin, RESTObject):
    _id_attr = None