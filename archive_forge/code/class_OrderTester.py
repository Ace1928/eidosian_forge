from typing import Any
from cirq.testing.equals_tester import EqualsTester
class OrderTester:
    """Tests ordering against user-provided disjoint ordered groups or items."""

    def __init__(self):
        self._groups = []
        self._eq_tester = EqualsTester()

    def _verify_ordering_one_sided(self, a: Any, b: Any, sign: int):
        """Checks that (a vs b) == (0 vs sign)."""
        for cmp_name, cmp_func in _NAMED_COMPARISON_OPERATORS:
            expected = cmp_func(0, sign)
            actual = cmp_func(a, b)
            assert expected == actual, f'Ordering constraint violated. Expected X={a} to {['be more than', 'equal', 'be less than'][sign + 1]} Y={b}, but X {cmp_name} Y returned {actual}'

    def _verify_ordering(self, a: Any, b: Any, sign: int):
        """Checks that (a vs b) == (0 vs sign) and (b vs a) == (sign vs 0)."""
        self._verify_ordering_one_sided(a, b, sign)
        self._verify_ordering_one_sided(b, a, -sign)

    def _verify_not_implemented_vs_unknown(self, item: Any):
        try:
            self._verify_ordering(_SmallerThanEverythingElse(), item, +1)
            self._verify_ordering(_EqualToEverything(), item, 0)
            self._verify_ordering(_LargerThanEverythingElse(), item, -1)
        except AssertionError as ex:
            raise AssertionError(f'Objects should return NotImplemented when compared to an unknown value, i.e. comparison methods should start with\n\n    if not isinstance(other, type(self)):\n        return NotImplemented\n\nThat rule is being violated by this value: {item!r}') from ex

    def add_ascending(self, *items: Any):
        """Tries to add a sequence of ascending items to the order tester.

        This methods asserts that items must all be ascending
        with regard to both each other and the elements which have been already
        added during previous calls.
        Some of the previously added elements might be equivalence groups,
        which are supposed to be equal to each other within that group.

        Args:
          *items: The sequence of strictly ascending items.

        Raises:
            AssertionError: Items are not ascending either
                with regard to each other, or with regard to the elements
                which have been added before.
        """
        for item in items:
            self.add_ascending_equivalence_group(item)

    def add_ascending_equivalence_group(self, *group_items: Any):
        """Tries to add an ascending equivalence group to the order tester.

        Asserts that the group items are equal to each other, but strictly
        ascending with regard to the already added groups.

        Adds the objects as a group.

        Args:
            *group_items: items making the equivalence group

        Raises:
            AssertionError: The group elements aren't equal to each other,
                or items in another group overlap with the new group.
        """
        for item in group_items:
            self._verify_not_implemented_vs_unknown(item)
        for item1 in group_items:
            for item2 in group_items:
                self._verify_ordering(item1, item2, 0)
        for lesser_group in self._groups:
            for lesser_item in lesser_group:
                for larger_item in group_items:
                    self._verify_ordering(lesser_item, larger_item, +1)
        self._eq_tester.add_equality_group(*group_items)
        self._groups.append(group_items)