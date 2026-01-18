import pytest
import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator
import sympy
def _assert_about_equal(actual: float, expected: float):
    """Assert that two times are within 25% of the expected time.

    Used to test the estimator with noisy data from actual devices.
    """
    assert expected * 0.75 < actual < expected * 1.25