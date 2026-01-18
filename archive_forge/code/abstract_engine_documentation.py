import abc
import datetime
from typing import Dict, List, Optional, Sequence, Set, Union
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine import abstract_job, abstract_program, abstract_processor
Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
        