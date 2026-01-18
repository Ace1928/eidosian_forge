from typing import Any, List, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject, RESTObjectList
class LDAPGroup(RESTObject):
    _id_attr = None