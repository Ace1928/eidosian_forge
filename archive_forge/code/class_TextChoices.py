import enum
import warnings
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import Promise
from django.utils.version import PY311, PY312
class TextChoices(Choices, StrEnum):
    """Class for creating enumerated string choices."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name