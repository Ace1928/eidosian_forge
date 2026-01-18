import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
@staticmethod
def _split_ranges(cnr, new_step):
    """Split a discrete range into a list of ranges using a new step.

        This takes a single NumericRange and splits it into a set
        of new ranges, all of which use a new step.  The new_step must
        be a multiple of the current step.  CNR objects with a step of 0
        are returned unchanged.

        Parameters
        ----------
            cnr: `NumericRange`
                The range to split
            new_step: `int`
                The new step to use for returned ranges

        """
    if cnr.step == 0 or new_step == 0:
        return [cnr]
    assert new_step >= abs(cnr.step)
    assert new_step % cnr.step == 0
    _dir = int(math.copysign(1, cnr.step))
    _subranges = []
    for i in range(int(abs(new_step // cnr.step))):
        if _dir * (cnr.start + i * cnr.step) > _dir * cnr.end:
            break
        _subranges.append(NumericRange(cnr.start + i * cnr.step, cnr.end, _dir * new_step))
    return _subranges