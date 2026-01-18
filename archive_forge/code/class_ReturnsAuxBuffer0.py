from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
class ReturnsAuxBuffer0:

    def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
        return args.auxiliary_buffer0