import copy
import os_service_types.data
from os_service_types import exc
def _canonical_project_name(self, name):
    """Convert repo name to project name."""
    if name is None:
        raise ValueError('Empty project name is not allowed')
    return name.rpartition('/')[-1]