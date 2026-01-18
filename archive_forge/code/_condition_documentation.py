import functools
import os
import unittest
Decorator that imposes the test to be successful at least once.

    Decorated test case is launched multiple times.
    The case is regarded as passed if it is successful
    at least once.

    .. note::
        In current implementation, this decorator grasps the
        failure information of each trial.

    Args:
        times(int): The number of trials.
    