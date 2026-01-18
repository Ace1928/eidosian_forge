from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
class ProjectArtifact(RESTObject):
    """Dummy object to manage custom actions on artifacts"""
    _id_attr = 'ref_name'