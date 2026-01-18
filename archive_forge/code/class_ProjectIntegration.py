from typing import Any, cast, List, Union
from gitlab import cli
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectIntegration(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'slug'