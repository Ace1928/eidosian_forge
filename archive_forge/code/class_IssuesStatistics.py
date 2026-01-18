from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class IssuesStatistics(RefreshMixin, RESTObject):
    _id_attr = None