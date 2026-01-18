from __future__ import annotations
from abc import ABC
from qiskit.providers import Options
class BasePrimitive(ABC):
    """Primitive abstract base class."""

    def __init__(self, options: dict | None=None):
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._run_options

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._run_options.update_options(**fields)