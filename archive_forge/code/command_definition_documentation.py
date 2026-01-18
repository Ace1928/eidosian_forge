from typing import (
from .constants import (
from .exceptions import (
from .utils import (

        Convenience method for removing a settable parameter from the CommandSet

        :param name: name of the settable being removed
        :raises: KeyError if the Settable matches this name
        