from typing import Any
from cirq.testing.equals_tester import EqualsTester
class _SmallerThanEverythingElse:

    def __eq__(self, other) -> bool:
        return isinstance(other, _SmallerThanEverythingElse)

    def __ne__(self, other) -> bool:
        return not isinstance(other, _SmallerThanEverythingElse)

    def __lt__(self, other) -> bool:
        return not isinstance(other, _SmallerThanEverythingElse)

    def __le__(self, other) -> bool:
        return True

    def __gt__(self, other) -> bool:
        return False

    def __ge__(self, other) -> bool:
        return isinstance(other, _SmallerThanEverythingElse)

    def __repr__(self) -> str:
        return 'SmallerThanEverythingElse'