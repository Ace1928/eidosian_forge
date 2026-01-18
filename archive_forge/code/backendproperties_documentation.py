import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix

        Given a SI unit prefix and value, apply the prefix to convert to
        standard SI unit.

        Args:
            value: The number to apply prefix to.
            unit: String prefix.

        Returns:
            Converted value.

        Raises:
            BackendPropertyError: If the units aren't recognized.
        