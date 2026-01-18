import abc
from typing import Dict, Iterator, List, Optional, overload, Sequence, Tuple, TYPE_CHECKING
import duet
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.engine_result import EngineResult
Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        