from typing import Any
from cirq.testing.equals_tester import EqualsTester
def _verify_ordering_one_sided(self, a: Any, b: Any, sign: int):
    """Checks that (a vs b) == (0 vs sign)."""
    for cmp_name, cmp_func in _NAMED_COMPARISON_OPERATORS:
        expected = cmp_func(0, sign)
        actual = cmp_func(a, b)
        assert expected == actual, f'Ordering constraint violated. Expected X={a} to {['be more than', 'equal', 'be less than'][sign + 1]} Y={b}, but X {cmp_name} Y returned {actual}'