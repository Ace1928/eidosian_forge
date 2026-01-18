import typing
from typing import Awaitable, Callable, Any, List

ValidationRule = Callable[[Any], Awaitable[bool]]


async def is_valid_example(value) -> Awaitable[bool]:
    """
    This is an example of a validation function.
    All validation functions must be awaitable, take any single value (can be another function for nesting etc.) and produce a boolean as output indicating pass/fail.
    Always ensure the validation is robust and in the case of error it raises the error and continues ratehr than stops.
    This is to ensure that all errors are captured and reported back to the user.
    """
    return True


# Actual Validators for This Module - __name___validators.py
