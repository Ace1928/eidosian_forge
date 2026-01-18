from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
class HasApplyMixture:

    def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
        args.target_tensor = 0.5 * args.target_tensor + 0.5 * np.dot(np.dot(x, args.target_tensor), x)
        return args.target_tensor