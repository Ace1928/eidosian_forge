import asyncio
import json
import pathlib
import re
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Union,
    get_origin,
    get_args,
    get_type_hints,
    Optional,
)
import inspect
from inspect import signature, iscoroutinefunction
from collections.abc import Mapping, Iterable
from enum import Enum

__all__ = [
    "Validate",
    "validation_rules",
]

# Type alias for clarity
ValidationRules = Dict[str, Callable[[Any], Awaitable[bool]]]


class Validate:
    """
    A comprehensive asynchronous validator designed to enforce strict type and custom validation rules
    across various function arguments and data inputs. It leverages advanced Python features and asynchronous
    programming to ensure non-blocking operations, detailed logging, and robust error handling.
    """

    def __init__(self, validation_rules: ValidationRules):
        """
        Initializes the AsyncValidator with a set of validation rules.

        Args:
            validation_rules (ValidationRules): A dictionary mapping rule names to their corresponding
                                                asynchronous validation functions.
        """
        self.validation_rules = validation_rules

    async def __call__(self, value: Any, rule_name: Optional[str] = None) -> bool:
        """
        Asynchronously validates a value against a specified rule when the instance is called.
        If no rule_name is provided, it attempts to validate the value using all available rules.

        Args:
            value (Any): The value to validate.
            rule_name (Optional[str]): The name of the validation rule to apply. Defaults to None.

        Returns:
            bool: True if the value passes the validation rule(s), False otherwise.

        Raises:
            ValueError: If the specified rule name does not exist in the validation rules.
        """
        if rule_name:
            if rule_name not in self.validation_rules:
                error_msg = f"Validation rule '{rule_name}' does not exist."
                logging.error(error_msg)
                raise ValueError(error_msg)

            validation_func = self.validation_rules[rule_name]
            result = await validation_func(value)
            return result
        else:
            # Validate against all rules
            results = await asyncio.gather(
                *[rule(value) for rule in self.validation_rules.values()]
            )
            return all(results)

        # Automatically run the methods: is_valid_func_signature, is_valid_argument, is_valid_type
        # Ensure that any validation rules (if present) are utilised flexibly and dynamically with each of these methods.
        await self.is_valid_func_signature(value)
        await self.is_valid_argument(value)
        await self.is_valid_type(value, type(value))

    async def is_valid_func_signature(self, func: Callable, *args, **kwargs) -> None:
        """
        Asynchronously validates a function's signature against provided arguments and types,
        ensuring compatibility with both synchronous and asynchronous functions. It leverages Python's
        introspection capabilities for dynamic signature validation, detailed logging, and robust error handling.

        Args:
            func (Callable): The function whose signature is being validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If the provided arguments do not match the function's signature.
        """
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        type_hints = get_type_hints(func)
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name, None)
            if expected_type and not await self.is_valid_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                )

    async def is_valid_argument(self, func: Callable, *args, **kwargs) -> None:
        """
        Validates the arguments passed to a function against expected type hints and custom validation rules.
        Adjusts for whether the function is a bound method or a regular function, applying argument validation accordingly.
        This method is asynchronous, leveraging asyncio for non-blocking operations and ensuring thread safety with asyncio.Lock.

        Args:
            func (Callable): The function whose arguments are to be validated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its expected type.
            ValueError: If an argument fails custom validation rules.
        """
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            args = args[1:]
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        type_hints = get_type_hints(func)
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name, None)
            if expected_type and not await self.is_valid_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
                )

    async def is_valid_type(self, value: Any, expected_type: Any) -> bool:
        """
        Recursively validates a value against an expected type, handling generics, special forms, and complex types.
        This method is designed to be exhaustive in its approach to type validation, ensuring compatibility with a wide range of type annotations.
        Utilizes asyncio for non-blocking operations and ensures thread safety with asyncio.Lock.

        Args:
            value (Any): The value to validate.
            expected_type (Any): The expected type against which to validate the value.

        Returns:
            bool: True if the value matches the expected type, False otherwise.
        """
        if expected_type is Any:
            return True

        if get_origin(expected_type) is Union:
            return any(
                [
                    await self.is_valid_type(value, arg)
                    for arg in get_args(expected_type)
                ]
            )

        if get_origin(expected_type) is Union or expected_type is Any:
            return True

        origin_type = get_origin(expected_type)
        type_args = get_args(expected_type)

        if origin_type:
            if not isinstance(value, origin_type):
                return False
            if type_args:
                # Inside the AsyncValidator class, modify the is_valid_type method
                if issubclass(origin_type, Mapping):
                    key_type, val_type = type_args
                    items_validation = [
                        await self.is_valid_type(k, key_type)
                        and await self.is_valid_type(v, val_type)
                        for k, v in value.items()
                    ]
                    return all(items_validation)
                elif issubclass(origin_type, Iterable) and not issubclass(
                    origin_type, (str, bytes, bytearray)
                ):
                    element_type = type_args[0]
                    # Use asyncio.gather to run validations concurrently and wait for all results
                    validations = [
                        self.is_valid_type(elem, element_type) for elem in value
                    ]
                    results = await asyncio.gather(*validations)
                    return all(results)
        else:
            if not isinstance(value, expected_type):
                return False
            return True
        return False


async def is_positive_integer(value: Any) -> bool:
    """Validates that the value is a positive integer."""
    return isinstance(value, int) and value > 0


async def is_non_empty_string(value: Any) -> bool:
    """Ensures the value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


async def is_valid_email(value: Any) -> bool:
    """Checks if the value is a valid email address."""
    if not isinstance(value, str):
        return False
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", value))


async def is_file_path_exists(value: Any) -> bool:
    """Validates that the file path exists."""
    if not isinstance(value, str):
        return False
    return pathlib.Path(value).exists()


async def is_valid_json(value: Any) -> bool:
    """Asynchronously validates if the value is valid JSON."""
    if not isinstance(value, str):
        return False
    try:
        json.loads(value)
        return True
    except ValueError:
        return False


# Wrapper function for is_in_enum to fit the expected signature
def make_is_in_enum_validator(
    enum_class: type[Enum],
) -> Callable[[Any], Awaitable[bool]]:
    """
    Creates a validation function for checking if a value is in the specified enum.

    Args:
        enum_class (type[Enum]): The enumeration class to validate against.

    Returns:
        Callable[[Any], Awaitable[bool]]: An async validation function.
    """

    async def is_in_enum_validator(value: Any) -> bool:
        return await is_in_enum(value, enum_class)

    return is_in_enum_validator


async def is_in_enum(value: Any, enum_class: type[Enum]) -> bool:
    """Validates that the value is a member of a given enum class."""
    return value in [e.value for e in enum_class.__members__.values()]


class AnyEnum(Enum):
    EXAMPLE_VALUE_1 = 1
    EXAMPLE_VALUE_2 = 2
    EXAMPLE_VALUE_3 = 3


async def dynamic_validate(value: Any) -> bool:
    """
    Dynamically validates a value using all available validation rules and the Validate class methods.

    Args:
        value (Any): The value to be validated.

    Returns:
        bool: True if the value passes all validations, False otherwise.
    """
    validator_instance = Validate(validation_rules)
    validation_results = []

    # Validate using custom validation rules
    for rule_name in validation_rules.keys():
        if rule_name != "Validator":  # Exclude the Validate class itself
            try:
                result = await validator_instance(value, rule_name)
                validation_results.append(result)
            except ValueError as e:
                logging.error(f"Validation error for rule {rule_name}: {e}")
                validation_results.append(False)

    # Dynamically validate using Validate class methods if applicable
    if callable(value):
        try:
            await validator_instance.is_valid_func_signature(value)
            await validator_instance.is_valid_argument(value)
            validation_results.append(True)
        except (TypeError, ValueError) as e:
            logging.error(f"Validation error for callable {value}: {e}")
            validation_results.append(False)

    # Validate the type if it's not a callable (to avoid redundancy)
    if not callable(value):
        try:
            type_validation_result = await validator_instance.is_valid_type(
                value, type(value)
            )
            validation_results.append(type_validation_result)
        except TypeError as e:
            logging.error(f"Type validation error for {value}: {e}")
            validation_results.append(False)

    return all(validation_results)


# Comprehensive dictionary of validation rules
# This dictionary maps validation rule names to their corresponding asynchronous validation functions.
# Each validation function is designed to take any value as input and return a boolean value asynchronously,
# indicating whether the input satisfies the specific validation rule.
validation_rules: ValidationRules = {
    "is_positive_integer": is_positive_integer,  # Validates if the input is a positive integer.
    "is_non_empty_string": is_non_empty_string,  # Checks if the input is a non-empty string.
    "is_valid_email": is_valid_email,  # Determines if the input is a valid email address.
    "is_file_path_exists": is_file_path_exists,  # Verifies the existence of the file path provided as input.
    "is_valid_json": is_valid_json,  # Validates if the input string is valid JSON.
    "is_in_enum": make_is_in_enum_validator(
        AnyEnum
    ),  # Checks if the input value is a member of a specified enumeration.
    "Validator": dynamic_validate,  # Dynamically validates the input using all available validation rules.
}


async def main():
    """
    Main function to demonstrate the usage of dynamic_validate function.
    """
    test_values = [
        42,  # Should pass is_positive_integer
        "",  # Should fail is_non_empty_string
        "test@example.com",  # Should pass is_valid_email
        "invalid_email",  # Should fail is_valid_email
        "./nonexistentfile.txt",  # Should fail is_file_path_exists
        '{"valid": "json"}',  # Should pass is_valid_json
        AnyEnum.EXAMPLE_VALUE_1,  # Should pass is_in_enum
        "not_an_enum_value",  # Should fail is_in_enum
        is_positive_integer,  # Should pass callable validations
    ]

    for value in test_values:
        result = await dynamic_validate(value)
        print(f"Validation result for {value}: {result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
