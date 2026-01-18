from enum import Enum, auto
from matplotlib import _docstring
class _AutoStringNameEnum(Enum):
    """Automate the ``name = 'name'`` part of making a (str, Enum)."""

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __hash__(self):
        return str(self).__hash__()