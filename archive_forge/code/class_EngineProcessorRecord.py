import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
@dataclasses.dataclass(frozen=True)
class EngineProcessorRecord(ProcessorRecord):
    """A serializable record of processor_id to map to a `cg.EngineProcessor`.

    This class presumes the GOOGLE_CLOUD_PROJECT environment
    variable is set to establish a connection to the cloud service.

    Args:
        processor_id: The processor id.
    """
    processor_id: str

    def get_processor(self) -> 'cg.EngineProcessor':
        """Return a `cg.EngineProcessor` for the specified processor_id."""
        engine = cg.get_engine()
        return engine.get_processor(self.processor_id)

    def __repr__(self):
        return dataclass_repr(self, namespace='cirq_google')

    def __str__(self) -> str:
        return self.processor_id

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)