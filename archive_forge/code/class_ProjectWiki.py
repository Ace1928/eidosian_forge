from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin, UploadMixin
from gitlab.types import RequiredOptional
class ProjectWiki(SaveMixin, ObjectDeleteMixin, UploadMixin, RESTObject):
    _id_attr = 'slug'
    _repr_attr = 'slug'
    _upload_path = '/projects/{project_id}/wikis/attachments'