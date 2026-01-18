"""
Module: working_document.py
Description: Contains the StandardDecorator class and a test suite for validating its functionality across various scenarios.
The StandardDecorator is designed to enhance functions with advanced features such as logging, error handling, performance monitoring, automatic retrying, result caching, and input validation.
This module serves as a test bed for ensuring the decorator's effectiveness and adaptability across different use cases within the EVIE project.
"""

# Importing necessary libraries with detailed descriptions
import collections  # For ordered dictionary support in caching mechanism
import logging  # For logging support
import asyncio  # For handling asynchronous operations
import functools  # For higher-order functions and operations on callable objects
import time  # For measuring execution time and implementing delays
from inspect import signature  # For inspecting callable objects and binding arguments
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Optional,
    Type,
    get_type_hints,
)  # For type hinting and annotations

# Type variable F, bound to Callable, for generic function annotations
F = TypeVar("F", bound=Callable[..., Any])


class StandardDecorator:
    """
    A class encapsulating a decorator that enhances functions with logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching,
    and dynamic input validation and sanitization. Designed for use across the Neuro Forge INDEGO (EVIE) project.

    This decorator provides a robust framework for enhancing function execution with features like
    automatic retries on specified exceptions, performance logging, input validation, and result caching
    with a thread-safe LRU cache strategy. It allows for granular control over logging levels and
    includes support for complex validation rules, making it highly customizable and adaptable to various needs.

    Example usage:
        @StandardDecorator(retries=2, delay=1, log_level=logging.INFO, cache_maxsize=100)
        def my_function(param1):
            # Function body

    Attributes:
        retries (int): The number of retry attempts for the decorated function upon failure.
        delay (int): The delay (in seconds) between retry attempts.
        cache_results (bool): Flag to enable or disable result caching for the decorated function.
        log_level (int): The logging level to be used for logging messages.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): A dictionary mapping argument names to validation functions.
        retry_exceptions (Tuple[Type[BaseException], ...]): A tuple of exception types that should trigger a retry.
        cache_maxsize (int): The maximum size of the cache (number of items) when result caching is enabled.
        enable_performance_logging (bool): Flag to enable or disable performance logging.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        cache_results: bool = False,
        log_level: int = logging.DEBUG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 128,
        enable_performance_logging: bool = True,
    ):
        """
        Initializes the StandardDecorator with the provided configuration parameters.
        """
        self.retries = retries
        self.delay = delay
        self.cache_results = cache_results
        self.log_level = log_level
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.cache_maxsize = cache_maxsize
        self.enable_performance_logging = enable_performance_logging
        self.cache = collections.OrderedDict()

    def cache_logic(self, key: Tuple[Any, ...], func: F, *args, **kwargs) -> Any:
        """
        Handles the caching logic for the decorated function, including cache hits and maintaining cache size.

        Args:
            key (Tuple[Any, ...]): The key under which the result is stored in the cache.
            func (F): The function to be executed and potentially cached.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution, either retrieved from the cache or newly computed.
        """
        if key in self.cache:
            logging.debug(
                f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            return self.cache[key]
        else:
            if self.cache_results:
                # Ensure the cache does not exceed the maximum size
                if len(self.cache) >= self.cache_maxsize:
                    self.cache.popitem(last=False)  # Remove the oldest item
                result = func(*args, **kwargs)
                self.cache[key] = result
                return result
            else:
                return func(*args, **kwargs)

    def __call__(self, func: F) -> F:
        """
        Makes the StandardDecorator class callable, allowing it to be used as a decorator.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The wrapped function, with enhanced functionality.
        """
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                return await self.wrapper_logic(func, True, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                return self.wrapper_logic(func, False, *args, **kwargs)

            return sync_wrapper

    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the core logic for retrying, caching, and logging the execution of the decorated function.

        Args:
            func (F): The function to be executed.
            is_async (bool): Indicates whether the function is asynchronous.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution.
        """
        self.validate_inputs(func, *args, **kwargs)
        key = (func.__name__, args, tuple(sorted(kwargs.items())))
        if self.cache_results and key in self.cache:
            return self.cache_logic(key, func, *args, **kwargs)
        attempts = 0
        while attempts < self.retries:
            try:
                start_time = time.perf_counter()
                result = (
                    await func(*args, **kwargs) if is_async else func(*args, **kwargs)
                )
                end_time = time.perf_counter()
                if self.enable_performance_logging:
                    logging.debug(
                        f"{func.__name__} executed in {end_time - start_time:.6f}s"
                    )
                if self.cache_results:
                    self.cache_logic(key, func, *args, **kwargs)  # Cache the result
                return result
            except self.retry_exceptions as e:
                logging.error(
                    f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                )
                attempts += 1
                if is_async:
                    await asyncio.sleep(self.delay)
                else:
                    time.sleep(self.delay)
        logging.debug(f"Final attempt for {func.__name__}")
        # Making a final attempt to execute the function after all retries have been exhausted
        return await func(*args, **kwargs) if is_async else func(*args, **kwargs)

    def validate_inputs(self, func: F, *args: Any, **kwargs: Any) -> None:
        """
        Validates the inputs to the decorated function based on type hints and custom validation rules.

        Args:
            func (F): The function being decorated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Raises:
            TypeError: If an argument does not match its type hint.
            ValueError: If an argument fails a custom validation rule.
        """
        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        arg_types = get_type_hints(func)

        for arg, value in bound_arguments.arguments.items():
            if arg in arg_types and not isinstance(value, arg_types[arg]):
                raise TypeError(f"Argument {arg} must be of type {arg_types[arg]}")
            if arg in self.validation_rules and not self.validation_rules[arg](value):
                raise ValueError(
                    f"Validation failed for argument {arg} with value {value}"
                )


if __name__ == "__main__":
    # Configure logging to ensure that log messages are displayed during the tests
    logging.basicConfig(level=logging.DEBUG)

    # Test 1: Synchronous function with input validation
    @StandardDecorator(
        retries=2,
        delay=1,
        log_level=logging.INFO,
        validation_rules={"x": lambda x: x > 0},  # Validation rule: x must be positive
    )
    def sync_example(x: int) -> int:
        """Synchronous test function that raises a ValueError for specific input to test retries."""
        if x == 5:
            raise ValueError("Example of a retry scenario.")
        return x * 2

    # Test 2: Asynchronous function with result caching
    @StandardDecorator(
        cache_results=True, cache_maxsize=2, enable_performance_logging=True
    )
    async def async_example(x: int) -> int:
        """Asynchronous test function that simulates an I/O operation to test caching."""
        await asyncio.sleep(1)  # Simulate an I/O operation
        return x**2

    # Test 3: Complex synchronous function demonstrating validation and error handling
    @StandardDecorator(
        retries=1,
        delay=2,
        log_level=logging.DEBUG,
        validation_rules={
            "text": lambda t: isinstance(t, str)
        },  # Validation rule: text must be a string
    )
    def complex_sync_example(text: str, repeat: int) -> str:
        """Complex synchronous function to test validation and error handling."""
        if repeat < 1:
            raise ValueError("repeat must be greater than 0")
        return text * repeat

    # Test 4: Complex asynchronous function to test retry and caching with transient failures
    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    async def complex_async_example(x: int) -> int:
        """Complex asynchronous function that simulates transient failures for even numbers."""
        if x % 2 == 0:
            raise Exception("Even numbers simulate transient failures.")
        await asyncio.sleep(2)  # Simulate a longer I/O operation
        return x + 10

    # Running synchronous examples
    try:
        print(f"Test 1 Result: {sync_example(5)}")
    except Exception as e:
        print(f"Test 1 Failed: {e}")

    print(f"Test 1 Repeat Result: {sync_example(10)}")

    try:
        print(f"Test 3 Result: {complex_sync_example('Test', 0)}")
    except Exception as e:
        print(f"Test 3 Failed: {e}")

    print(f"Test 3 Repeat Result: {complex_sync_example('Repeat', 3)}")

    # Running asynchronous examples
    # Test 4: Complex asynchronous function to test retry and caching with transient failures
    @StandardDecorator(
        retries=3, delay=1, cache_results=True, enable_performance_logging=True
    )
    async def run_async_examples():
        try:
            print(f"Test 2 Result: {await async_example(3)}")
            print(
                f"Test 2 Cache Hit: {await async_example(3)}"
            )  # This should hit the cache
        except Exception as e:
            print(f"Test 2 Failed: {e}")

        try:
            print(f"Test 4 Result: {await complex_async_example(4)}")
        except Exception as e:
            print(f"Test 4 Failed: {e}")

        print(f"Test 4 Repeat Result: {await complex_async_example(5)}")

    asyncio.run(run_async_examples())

"""
TODO:
- Expand the test suite to cover more edge cases and input scenarios.
- Consider integrating a more sophisticated logging mechanism that can capture and segregate logs based on test cases for easier analysis.
- Implement additional validation rules in the test functions to thoroughly test the dynamic input validation feature of the StandardDecorator.
- Explore the possibility of adding a feature to the decorator for automatic documentation of test cases and outcomes.
- Investigate the integration of a continuous integration (CI) pipeline to automatically run these tests upon each commit to ensure code quality and functionality.
- Enhance error handling in test cases to include more specific exception types, providing clearer insights into failure points.
- Evaluate the performance impact of the decorator on function execution times in real-world scenarios to ensure it meets project requirements.
- Test the decorator's compatibility with different Python versions and environments to ensure broad usability across the EVIE project.
- Develop a mechanism for dynamically adjusting log levels and retry parameters based on environmental variables or configuration files.
- Consider adding support for asynchronous validation rules to accommodate functions that require I/O operations for input validation.

KNOWN ISSUES:
- The current caching mechanism does not differentiate between function calls with keyword arguments in different orders; this could lead to unexpected cache hits or misses.
- Validation rules are limited to synchronous functions; asynchronous validation is not currently supported but is a planned feature.
- Performance logging does not account for the overhead introduced by the decorator itself, which may skew metrics slightly.

ADDITIONAL FUNCTIONALITIES:
- Implement a feature for the decorator to automatically generate a report of the test suite's execution, including success rates, performance metrics, and encountered errors.
- Explore the integration of a visual dashboard to display real-time results of the test suite executions, enhancing visibility for the development team.
- Add support for parallel execution of test cases to reduce the total runtime of the test suite, especially beneficial for projects with a large number of tests.
"""
