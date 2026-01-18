import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
def _get_channel_prefix_index(self, channel: str) -> str:
    """Return channel prefix and index from the given ``channel``.

        Args:
            channel: Name of channel.

        Raises:
            BackendConfigurationError: If invalid channel name is found.

        Return:
            Channel name and index. For example, if ``channel=acquire0``, this method
            returns ``acquire`` and ``0``.
        """
    channel_prefix = re.match('(?P<channel>[a-z]+)(?P<index>[0-9]+)', channel)
    try:
        return (channel_prefix.group('channel'), int(channel_prefix.group('index')))
    except AttributeError as ex:
        raise BackendConfigurationError(f"Invalid channel name - '{channel}' found.") from ex