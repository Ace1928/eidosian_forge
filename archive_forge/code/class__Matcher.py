from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
class _Matcher:
    """A class for making assertions on expectation values."""

    def __init__(self, client: ls_client.Client, key: str, value: Any, _executor: Optional[concurrent.futures.ThreadPoolExecutor]=None, run_id: Optional[str]=None):
        self.client = client
        self.key = key
        self.value = value
        self._executor = _executor or concurrent.futures.ThreadPoolExecutor(max_workers=3)
        rt = rh.get_current_run_tree()
        self._run_id = rt.id if rt else run_id

    def _submit_feedback(self, score: int, message: Optional[str]=None) -> None:
        if not ls_utils.test_tracking_is_disabled():
            self._executor.submit(self.client.create_feedback, run_id=self._run_id, key='expectation', score=score, comment=message)

    def _assert(self, condition: bool, message: str, method_name: str) -> None:
        try:
            assert condition, message
            self._submit_feedback(1, message=f'Success: {self.key}.{method_name}')
        except AssertionError as e:
            self._submit_feedback(0, repr(e))
            raise e from None

    def to_be_less_than(self, value: float) -> None:
        """Assert that the expectation value is less than the given value.

        Args:
            value: The value to compare against.

        Raises:
            AssertionError: If the expectation value is not less than the given value.
        """
        self._assert(self.value < value, f'Expected {self.key} to be less than {value}, but got {self.value}', 'to_be_less_than')

    def to_be_greater_than(self, value: float) -> None:
        """Assert that the expectation value is greater than the given value.

        Args:
            value: The value to compare against.

        Raises:
            AssertionError: If the expectation value is not
            greater than the given value.
        """
        self._assert(self.value > value, f'Expected {self.key} to be greater than {value}, but got {self.value}', 'to_be_greater_than')

    def to_be_between(self, min_value: float, max_value: float) -> None:
        """Assert that the expectation value is between the given min and max values.

        Args:
            min_value: The minimum value (exclusive).
            max_value: The maximum value (exclusive).

        Raises:
            AssertionError: If the expectation value
                is not between the given min and max.
        """
        self._assert(min_value < self.value < max_value, f'Expected {self.key} to be between {min_value} and {max_value}, but got {self.value}', 'to_be_between')

    def to_be_approximately(self, value: float, precision: int=2) -> None:
        """Assert that the expectation value is approximately equal to the given value.

        Args:
            value: The value to compare against.
            precision: The number of decimal places to round to for comparison.

        Raises:
            AssertionError: If the rounded expectation value
                does not equal the rounded given value.
        """
        self._assert(round(self.value, precision) == round(value, precision), f'Expected {self.key} to be approximately {value}, but got {self.value}', 'to_be_approximately')

    def to_equal(self, value: float) -> None:
        """Assert that the expectation value equals the given value.

        Args:
            value: The value to compare against.

        Raises:
            AssertionError: If the expectation value does
                not exactly equal the given value.
        """
        self._assert(self.value == value, f'Expected {self.key} to be equal to {value}, but got {self.value}', 'to_equal')

    def to_contain(self, value: Any) -> None:
        """Assert that the expectation value contains the given value.

        Args:
            value: The value to check for containment.

        Raises:
            AssertionError: If the expectation value does not contain the given value.
        """
        self._assert(value in self.value, f'Expected {self.key} to contain {value}, but it does not', 'to_contain')

    def against(self, func: Callable, /) -> None:
        """Assert the expectation value against a custom function.

        Args:
            func: A custom function that takes the expectation value as input.

        Raises:
            AssertionError: If the custom function returns False.
        """
        func_signature = inspect.signature(func)
        self._assert(func(self.value), f'Assertion {func_signature} failed for {self.key}', 'against')