import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
class ExecutableSpec(metaclass=abc.ABCMeta):
    """Specification metadata about an executable.

    Subclasses should add problem-specific fields.
    """
    executable_family: str = NotImplemented
    'A unique name to group executables.'