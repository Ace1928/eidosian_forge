from __future__ import absolute_import, print_function, unicode_literals
import typing
from typing import cast
import six
from copy import deepcopy
from ._typing import Text, overload
from .enums import ResourceType
from .errors import MissingInfoNamespace
from .path import join
from .permissions import Permissions
from .time import epoch_to_datetime
@property
def accessed(self):
    """`~datetime.datetime`: the resource last access time, or `None`.

        Requires the ``"details"`` namespace.

        Raises:
            ~fs.errors.MissingInfoNamespace: if the ``"details"``
                namespace is not in the Info.

        """
    self._require_namespace('details')
    _time = self._make_datetime(self.get('details', 'accessed'))
    return _time