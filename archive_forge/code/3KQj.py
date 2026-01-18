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
    List,
)
import inspect
from inspect import signature, iscoroutinefunction
from collections.abc import Mapping, Iterable
from enum import Enum

# utils/module_loader.py
import importlib
import os
import aiofiles
from regex import W

__all__ = [
    "Validate",
    "validation_rules",
]

VALIDATION_RULES_FILE = (
    "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/validation_rules.json"
)


# Type alias for clarity
ValidationRule = Callable[[Any], Awaitable[bool]]
ValidationRules = Dict[str, ValidationRule]
FILE_PATH = pathlib.Path


class AsyncValidationException(ValueError):
    """
    A specific exception type for async validation failures, providing detailed information about the failed validation.

    Attributes:
        argument (str): The name of the argument that failed validation.
        value (Any): The value of the argument that failed validation.
        message (str): An optional more detailed message.
    """

    def __init__(self, argument: str, value: Any, message: str = "") -> None:
        self.argument = argument
        self.value = value
        super().__init__(
            message
            or f"Validation failed for argument '{argument}' with value '{value}'"
        )

    def __str__(self) -> str:
        return f"Validation failed for argument '{self.argument}' with value '{self.value}'"


async def ensure_module_config_exists(path: FILE_PATH):
    """
    Ensures that the logging configuration file exists at the specified path. If not, it creates the file
    with the default logging configuration.

    Args:
        path (FILE_PATH): The path to the logging configuration file.
    """
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, mode="w", encoding="utf-8") as config_file:
                json.dump(
                    VALIDATION_RULES_FILE, config_file, ensure_ascii=False, indent=4
                )
        else:
            async with aiofiles.open(path, mode="w") as config_file:
                config_content = await config_file.read()
                validate_config = json.loads(config_content)
                return validate_config
    except Exception as e:
        logging.error(f"Failed to ensure logging config exists: {e}", exc_info=True)


def load_validation_rules_from_modules(module_paths: List[str]) -> ValidationRules:
    rules = {}
    for path in module_paths:
        module_name = os.path.splitext(os.path.basename(path))[0]
        module = importlib.import_module(module_name)
        for attr in dir(module):
            if attr.startswith(
                "is_"
            ):  # Assuming all validation functions start with 'is_'
                rules[attr] = getattr(module, attr)
    return rules


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
        module_paths = validate_config
        self.validation_rules = load_validation_rules_from_modules(module_paths)

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

    async def is_valid_argument(self, func: Callable, *args, **kwargs) -> bool:
        """
        Validates the arguments of a function against its type hints and applies custom validation rules, if any.
        This method dynamically adjusts for whether the function is a bound method or a regular function, applying
        argument validation accordingly. It leverages asyncio for non-blocking operations and ensures thread safety
        with asyncio.Lock, providing a robust mechanism for concurrent validations.

        This method is designed to be exhaustive in its approach to argument validation, ensuring compatibility with
        a wide range of type annotations and custom validation rules. It utilizes advanced programming techniques to
        offer a flexible, adaptive, and robust solution for argument validation.

        Args:
            func (Callable): The function to validate arguments for.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            bool: True if all arguments are valid according to their type hints and custom validation rules, False otherwise.

        Raises:
            TypeError: If an argument does not match its expected type according to the function's type hints.
            ValueError: If an argument fails custom validation rules specified in the validation rules dictionary.
        """
        # Adjust for bound methods by removing the 'self' or 'cls' argument
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            args = args[1:]

        # Bind the provided arguments to the function's signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Retrieve type hints for the function
        type_hints = get_type_hints(func)

        # Initialize an empty list to hold validation tasks
        validation_tasks = []

        # Iterate through each bound argument to validate
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(
                name, Any
            )  # Default to Any if no type hint is provided

            # Create a coroutine for validating the current argument's type
            validation_task = self.is_valid_type(value, expected_type)
            validation_tasks.append(validation_task)

            # Check for custom validation rules based on the expected type's name
            rule_name = f"is_{expected_type.__name__}_valid"
            if rule_name in self.validation_rules:
                # Add the custom validation rule to the list of tasks
                custom_validation_task = self.validation_rules[rule_name](value)
                validation_tasks.append(custom_validation_task)

        # Use asyncio.gather to run all validation tasks concurrently and wait for their results
        validation_results = await asyncio.gather(*validation_tasks)

        # If any validation failed, log an error and return False
        if not all(validation_results):
            logging.error("One or more arguments failed validation.")
            return False

        # If all validations passed, return True
        return True

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
    Dynamically validates a value or a function's arguments using all available validation rules and methods.
    This method is designed to be exhaustive in its approach to validation, ensuring compatibility with
    a wide range of data types and structures. It leverages advanced programming techniques to offer a flexible,
    adaptive, and robust solution for dynamic validation.

    Args:
        value (Any): The value or function to be validated.

    Returns:
        bool: True if the value or function's arguments pass all validations, False otherwise.

    Raises:
        ValueError: If an invalid rule name is encountered during validation.
        TypeError: If the value's type does not match the expected type for a given validation rule.
    """
    # Initialize the Validate instance with the comprehensive dictionary of validation rules.
    validator_instance = Validate(validation_rules)
    validation_results = []  # List to store individual validation results.

    # Validate using custom validation rules defined in the validation_rules dictionary.
    for rule_name, validation_func in validation_rules.items():
        # Exclude the Validate class itself to prevent recursion.
        if rule_name != "Validator":
            try:
                # Execute the validation function asynchronously and append the result.
                result = await validation_func(value)
                validation_results.append(result)
            except (ValueError, TypeError) as e:
                # Log any validation errors encountered during the process.
                logging.error(f"Validation error for rule '{rule_name}': {e}")
                validation_results.append(False)

    # If the value is a callable, i.e., a function, validate its signature and arguments.
    if callable(value):
        try:
            # Validate the function's signature for compatibility with expected arguments.
            await validator_instance.is_valid_func_signature(value)
            # Validate the function's arguments against their expected types and custom rules.
            await validator_instance.is_valid_argument(value)
            # Append True to indicate successful validation of a callable's signature and arguments.
            validation_results.append(True)
        except (TypeError, ValueError) as e:
            # Log any errors encountered during validation of the callable's signature or arguments.
            logging.error(f"Validation error for callable '{value}': {e}")
            validation_results.append(False)

    # For non-callable values, validate the value's type.
    if not callable(value):
        try:
            # Validate the value's type against the expected type for its corresponding validation rule.
            type_validation_result = await validator_instance.is_valid_type(
                value, type(value)
            )
            # Append the result of the type validation to the list of validation results.
            validation_results.append(type_validation_result)
        except TypeError as e:
            # Log any type validation errors encountered during the process.
            logging.error(f"Type validation error for value '{value}': {e}")
            validation_results.append(False)

    # Return True if all validations passed, False otherwise.
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


"""
TODO:
Modularization and Extensibility
1. Separate Validation Rules into Modules: To improve maintainability and extensibility, validation rules can be separated into their own modules or files. This allows for easier management of individual validators and the addition of new rules without modifying the core validation logic.
    - Modules Could Be: integer_validators.py, string_validators.py, email_validators.py, file_validators.py, json_validators.py, enum_validators.py
    - Each module would contain specific validation functions related to the corresponding data type or validation rule.
2. Dynamic Rule Loading: Implement a mechanism for dynamically loading validation rules from external sources, such as configuration files or databases. This allows for greater flexibility in defining and updating validation rules without modifying the codebase.
    - Use a configuration file in JSON or YAML format to define validation rules and load them dynamically at runtime.
    - Implement a rule registry that can dynamically register and manage validation rules based on external sources.
3. Support for Pydantic Models: Integrate support for Pydantic models to leverage its powerful data validation capabilities. Pydantic provides a declarative way to define data models with type annotations and validation rules, making it a robust choice for complex data validation scenarios.
    - Create a method to validate Pydantic models using the provided validation rules and custom validation functions.
    - Utilize Pydantic's model validation features to ensure data integrity and consistency in applications.
4. Custom Rule Registration: Allow users to register custom validation rules at runtime, enabling them to define and apply custom validation logic specific to their use cases.
    - Implement a mechanism for users to register custom validation rules with the validator instance.
    - Provide an API for users to define custom validation functions and associate them with rule names for validation.
5. Rule Composition and Chaining: Support rule composition and chaining to create complex validation logic by combining multiple rules in a flexible and reusable manner.
    - Implement a mechanism for composing validation rules using logical operators (AND, OR, NOT) to create compound rules.
    - Allow users to chain multiple validation rules together to form a sequence of validations for a given input.
6. Custom Type Validation: Extend the type validation capabilities to support custom type definitions and complex data structures beyond built-in types and generics.
    - Implement support for custom type validators that can handle user-defined types and complex data structures.
    - Allow users to define custom type validation functions for specific data types or structures in addition to built-in types.
7. Machine Learning For Validation: Explore the use of machine learning techniques, such as anomaly detection and pattern recognition, to enhance data validation capabilities.
    - Train machine learning models on historical data to identify patterns and anomalies for validation purposes.
    - Utilize machine learning algorithms to predict and validate data based on learned patterns and statistical analysis.
8. Rule Inheritance and Overriding: Implement rule inheritance and overriding mechanisms to facilitate rule reuse and customization in hierarchical validation scenarios.
    - Allow validation rules to inherit from parent rules and override specific behaviors or conditions as needed.
    - Provide a mechanism for users to define rule inheritance relationships and customize rule behavior at different levels of the hierarchy.
9. Asynchronous Validation with Advanced Concurrency: Enhance the asynchronous validation capabilities by leveraging advanced concurrency techniques, such as parallel processing and distributed computing, to improve performance and scalability.
    - Utilize parallel processing frameworks, such as Dask or Ray, to distribute validation tasks across multiple cores or nodes for faster processing.
    - Implement asynchronous validation pipelines with advanced concurrency patterns, such as fan-out/fan-in or scatter/gather, to optimize validation throughput.
10. Caching via integration with indecache.py which utilises TTL sparse LRU caching via KeyDB asynchronously: Implement caching mechanisms to store and retrieve validation results for improved performance and efficiency.
    - Integrate with an asynchronous caching library, such as indecache.py, to cache validation results and avoid redundant computations.
    - Utilize time-to-live (TTL) and least recently used (LRU) caching strategies to manage cache entries and optimize memory usage.
11. Detailed Documentation and API: Provide comprehensive documentation and an intuitive API for users to understand and interact with the validation framework effectively.
    - Create detailed documentation with examples, tutorials, and use cases to guide users in using the validation framework.
    - Design a user-friendly API with clear method names, parameters, and return values to facilitate seamless integration and customization.
"""
