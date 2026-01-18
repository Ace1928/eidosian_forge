import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class BiReservationReference(Reference):
    """Helper class to provide a reference to bi reservation."""
    _required_fields = frozenset(('projectId', 'location'))
    _format_str = '%(projectId)s:%(location)s'
    _path_str = 'projects/%(projectId)s/locations/%(location)s/biReservation'
    _create_path_str = 'projects/%(projectId)s/locations/%(location)s'
    typename = 'bi reservation'

    def path(self) -> str:
        return self._path_str % dict(self)

    def create_path(self) -> str:
        return self._create_path_str % dict(self)