from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
@no_type_check
def DELAY(*args) -> Union[DelayFrames, DelayQubits]:
    """
    Produce a DELAY instruction.

    Note: There are two variants of DELAY. One applies to specific frames on some
    qubit, e.g. `DELAY 0 "rf" "ff" 1.0` delays the `"rf"` and `"ff"` frames on 0.
    It is also possible to delay all frames on some qubits, e.g. `DELAY 0 1 2 1.0`.

    :param args: A list of delay targets, ending with a duration.
    :returns: A DelayFrames or DelayQubits instance.
    """
    if len(args) < 2:
        raise ValueError('Expected DELAY(t1,...,tn, duration). In particular, there must be at least one target, as well as a duration.')
    targets, duration = (args[:-1], args[-1])
    if not isinstance(duration, (Expression, Real)):
        raise TypeError('The last argument of DELAY must be a real or parametric duration.')
    if all((isinstance(t, Frame) for t in targets)):
        return DelayFrames(targets, duration)
    elif all((isinstance(t, (int, Qubit, FormalArgument)) for t in targets)):
        targets = [Qubit(t) if isinstance(t, int) else t for t in targets]
        return DelayQubits(targets, duration)
    else:
        raise TypeError(f'DELAY targets must be either (i) a list of frames, or (ii) a list of qubits / formal arguments. Got {args}.')