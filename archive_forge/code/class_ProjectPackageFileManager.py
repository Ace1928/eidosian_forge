from pathlib import Path
from typing import (
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, GetMixin, ListMixin, ObjectDeleteMixin
class ProjectPackageFileManager(DeleteMixin, ListMixin, RESTManager):
    _path = '/projects/{project_id}/packages/{package_id}/package_files'
    _obj_cls = ProjectPackageFile
    _from_parent_attrs = {'project_id': 'project_id', 'package_id': 'id'}