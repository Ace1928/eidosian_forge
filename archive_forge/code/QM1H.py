"""
Module: working_document.py
Description: Contains the StandardDecorator class and a test suite for validating its functionality across various scenarios.
The StandardDecorator is designed to enhance functions with advanced features such as logging, error handling, performance monitoring, automatic retrying, result caching, and input validation.
This module serves as a test bed for ensuring the decorator's effectiveness and adaptability across different use cases within the EVIE project.
"""

# Importing necessary libraries with detailed descriptions
import asyncio
import collections
import functools
import inspect
import logging
import time
from inspect import signature, get_type_hints, get_origin, get_args
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Optional,
    Type,
    Union,
    Awaitable,
    List,
)
import tracemalloc  # For tracking memory usage and identifying memory leaks

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
        dynamic_retry_enabled (bool): Flag to enable or disable the dynamic retry strategy.
        cache_key_strategy (Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]]): Custom function for generating cache keys.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        cache_results: bool = True,
        log_level: int = logging.INFO,
        validation_rules: Optional[
            Dict[str, Callable[[Any], Union[bool, Awaitable[bool]]]]
        ] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 128,
        enable_performance_logging: bool = True,
        dynamic_retry_enabled: bool = True,
        cache_key_strategy: Callable[
            [Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]
        ] = None,
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
        self.dynamic_retry_enabled = dynamic_retry_enabled
        self.cache_key_strategy = cache_key_strategy or self.generate_cache_key
        self.cache = collections.OrderedDict()
        logging.basicConfig(level=self.log_level)

    async def cache_logic(self, key: Tuple[Any, ...], func: F, *args, **kwargs) -> Any:
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
        key = self.cache_key_strategy(func, args, kwargs)
        if key in self.cache:
            logging.debug(
                f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            return self.cache[key]
        else:
            if self.cache_results:
                if len(self.cache) >= self.cache_maxsize:
                    self.cache.popitem(last=False)  # Remove the oldest item
                result = (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )
                self.cache[key] = result
                return result
            else:
                return (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

    async def invalidate_cache(
        self, condition: Callable[[Tuple[Any, ...], Any], bool]
    ) -> None:
        """
        Asynchronously invalidates cache entries based on a given condition.

        Args:
            condition (Callable[[Tuple[Any, ...], Any], bool]): A function that takes a cache key and value, returning True if the entry should be invalidated.
        """
        to_invalidate = [
            key for key, value in self.cache.items() if condition(key, value)
        ]
        for key in to_invalidate:
            del self.cache[key]

    def generate_cache_key(
        self, func: F, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """
        Generates a cache key that uniquely identifies a function call, taking into account the order of keyword arguments.

        Args:
            func (F): The function being called.
            args (Tuple[Any, ...]): Positional arguments of the function call.
            kwargs (Dict[str, Any]): Keyword arguments of the function call.

        Returns:
            Tuple[Any, ...]: A tuple representing the unique cache key.
        """
        kwargs_key = tuple(sorted(kwargs.items()))
        return (func.__name__, args, kwargs_key)

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
                # Check if there is an existing event loop and if it's running
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:  # No running event loop
                    loop = None

                if loop and loop.is_running():
                    # If there's a running event loop, use create_task to schedule the coroutine
                    # Note: This requires the caller to be in an async context or manage the event loop manually
                    future = asyncio.ensure_future(
                        self.wrapper_logic(func, False, *args, **kwargs)
                    )
                    return future
                else:
                    # No running event loop, safe to use asyncio.run
                    return asyncio.run(self.wrapper_logic(func, False, *args, **kwargs))

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
        await self.validate_async_rules(func, *args, **kwargs)
        self.validate_inputs(func, *args, **kwargs)
        key = self.cache_key_strategy(func, args, kwargs)
        if self.cache_results and key in self.cache:
            return await self.cache_logic(key, func, *args, **kwargs)
        attempts = 0
        while attempts < self.retries:
            try:
                start_time = time.perf_counter()
                result = (
                    await func(*args, **kwargs) if is_async else func(*args, **kwargs)
                )
                end_time = time.perf_counter()
                if self.enable_performance_logging:
                    self.log_performance(func, start_time, end_time)
                if self.cache_results:
                    await self.cache_logic(
                        key, func, *args, **kwargs
                    )  # Cache the result
                return result
            except self.retry_exceptions as e:
                if self.dynamic_retry_enabled:
                    dynamic_retries, dynamic_delay = self.dynamic_retry_strategy(e)
                    self.retries = (
                        dynamic_retries if dynamic_retries is not None else self.retries
                    )
                    self.delay = (
                        dynamic_delay if dynamic_delay is not None else self.delay
                    )
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

    async def validate_async_rules(self, func: F, *args: Any, **kwargs: Any) -> None:
        """
        Validates the inputs to the decorated function based on asynchronous validation rules.

        Args:
            func (F): The function being decorated.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.
        """
        bound_arguments = signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        for arg, value in bound_arguments.arguments.items():
            if arg in self.validation_rules:
                validation_rule = self.validation_rules[arg]
                if asyncio.iscoroutinefunction(validation_rule):
                    validation_result = await validation_rule(value)
                    if not validation_result:
                        raise ValueError(
                            f"Validation failed for argument {arg} with value {value}"
                        )
                else:
                    if not validation_rule(value):
                        raise ValueError(
                            f"Validation failed for argument {arg} with value {value}"
                        )

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
            expected_type = arg_types.get(arg)
            if expected_type is not None:
                # Check if the expected type is a complex type from the typing module
                if get_origin(expected_type) is not None:
                    # This is a complex type (e.g., List[int], Optional[str]), skip isinstance check
                    continue
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument {arg} must be of type {expected_type}, got type {type(value)}"
                    )
            if arg in self.validation_rules and not self.validation_rules[arg](value):
                raise ValueError(
                    f"Validation failed for argument {arg} with value {value}"
                )


if __name__ == "__main__":
    tracemalloc.start()
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

    # Running asynchronous examples in an async function to properly await them
    async def run_async_tests():
        try:
            result = await async_example(3)
            print(f"Test 2 Result: {result}")
            cache_hit_result = await async_example(3)  # This should hit the cache
            print(f"Test 2 Cache Hit: {cache_hit_result}")
        except Exception as e:
            print(f"Test 2 Failed: {e}")

        try:
            result = await complex_async_example(4)
            print(f"Test 4 Result: {result}")
        except Exception as e:
            print(f"Test 4 Failed: {e}")

        repeat_result = await complex_async_example(5)
        print(f"Test 4 Repeat Result: {repeat_result}")

    asyncio.run(run_async_tests())

"""
TODO:
- Consider adding support for asynchronous validation rules to accommodate functions that require I/O operations for input validation.

KNOWN ISSUES:
- The current caching mechanism does not differentiate between function calls with keyword arguments in different orders; this could lead to unexpected cache hits or misses.
- Validation rules are limited to synchronous functions; asynchronous validation is not currently supported but is a planned feature.
- Performance logging does not account for the overhead introduced by the decorator itself, which may skew metrics slightly.

ADDITIONAL FUNCTIONALITIES:
- Add support for parallel execution of test cases to reduce the total runtime of the test suite, especially beneficial for projects with a large number of tests.
"""
