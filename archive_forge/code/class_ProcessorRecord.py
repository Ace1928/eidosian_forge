import abc
import dataclasses
from typing import Any, Dict
import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
class ProcessorRecord(metaclass=abc.ABCMeta):
    """A serializable record that maps to a particular `cg.engine.AbstractProcessor`."""

    @abc.abstractmethod
    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        """Using this classes' attributes, return a unique `cg.engine.AbstractProcessor`

        This is the primary method that descendants must implement.
        """

    def get_sampler(self) -> 'cg.ProcessorSampler':
        """Return a `cirq.Sampler` for the processor specified by this class.

        The default implementation delegates to `self.get_processor()`.
        """
        return self.get_processor().get_sampler()

    def get_device(self) -> 'cirq.Device':
        """Return a `cirq.Device` for the processor specified by this class.

        The default implementation delegates to `self.get_processor()`.
        """
        return self.get_processor().get_device()