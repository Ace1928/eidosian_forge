import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class ReservationReference(Reference):
    _required_fields = frozenset(('projectId', 'location', 'reservationId'))
    _format_str = '%(projectId)s:%(location)s.%(reservationId)s'
    _path_str = 'projects/%(projectId)s/locations/%(location)s/reservations/%(reservationId)s'
    typename = 'reservation'

    def path(self) -> str:
        return self._path_str % dict(self)