from typing import Any
from cirq.testing.equals_tester import EqualsTester
def _verify_not_implemented_vs_unknown(self, item: Any):
    try:
        self._verify_ordering(_SmallerThanEverythingElse(), item, +1)
        self._verify_ordering(_EqualToEverything(), item, 0)
        self._verify_ordering(_LargerThanEverythingElse(), item, -1)
    except AssertionError as ex:
        raise AssertionError(f'Objects should return NotImplemented when compared to an unknown value, i.e. comparison methods should start with\n\n    if not isinstance(other, type(self)):\n        return NotImplemented\n\nThat rule is being violated by this value: {item!r}') from ex