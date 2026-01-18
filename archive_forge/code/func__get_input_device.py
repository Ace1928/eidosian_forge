import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
def _get_input_device(self) -> 'cirq.Device':
    """Return a `cg.GridDevice` for the specified processor_id.

        Only 'rainbow' and 'weber' are recognized processor_ids and the device information
        may not be up-to-date, as it is completely local.
        """
    device_spec = _create_device_spec_from_template(MOST_RECENT_TEMPLATES[self.processor_id])
    device = cg.GridDevice.from_proto(device_spec)
    return device