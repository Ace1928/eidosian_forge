from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
class NoContent(Exception):
    pass