from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
class ProjectResourceGroup(SaveMixin, RESTObject):
    _id_attr = 'key'
    upcoming_jobs: 'ProjectResourceGroupUpcomingJobManager'