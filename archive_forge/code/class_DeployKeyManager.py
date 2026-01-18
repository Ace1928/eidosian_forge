from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ListMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class DeployKeyManager(ListMixin, RESTManager):
    _path = '/deploy_keys'
    _obj_cls = DeployKey