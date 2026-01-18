import enum
import warnings
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import Promise
from django.utils.version import PY311, PY312
class IntegerChoices(Choices, IntEnum):
    """Class for creating enumerated integer choices."""
    pass