from __future__ import annotations
from flake8.formatting import base
from flake8.violation import Violation
class Pylint(SimpleFormatter):
    """Pylint formatter for Flake8."""
    error_format = '%(path)s:%(row)d: [%(code)s] %(text)s'