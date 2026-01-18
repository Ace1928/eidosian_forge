import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
Return channel prefix and index from the given ``channel``.

        Args:
            channel: Name of channel.

        Raises:
            BackendConfigurationError: If invalid channel name is found.

        Return:
            Channel name and index. For example, if ``channel=acquire0``, this method
            returns ``acquire`` and ``0``.
        