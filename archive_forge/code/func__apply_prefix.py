import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def _apply_prefix(self, value: float, unit: str) -> float:
    """
        Given a SI unit prefix and value, apply the prefix to convert to
        standard SI unit.

        Args:
            value: The number to apply prefix to.
            unit: String prefix.

        Returns:
            Converted value.

        Raises:
            BackendPropertyError: If the units aren't recognized.
        """
    try:
        return apply_prefix(value, unit)
    except Exception as ex:
        raise BackendPropertyError(f'Could not understand units: {unit}') from ex