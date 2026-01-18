import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class ReservationAssignmentReference(Reference):
    """Helper class to provide a reference to reservation assignment."""
    _required_fields = frozenset(('projectId', 'location', 'reservationId', 'reservationAssignmentId'))
    _format_str = '%(projectId)s:%(location)s.%(reservationId)s.%(reservationAssignmentId)s'
    _path_str = 'projects/%(projectId)s/locations/%(location)s/reservations/%(reservationId)s/assignments/%(reservationAssignmentId)s'
    _reservation_format_str = '%(projectId)s:%(location)s.%(reservationId)s'
    typename = 'reservation assignment'

    def path(self) -> str:
        return self._path_str % dict(self)

    def reservation_path(self) -> str:
        return self._reservation_format_str % dict(self)