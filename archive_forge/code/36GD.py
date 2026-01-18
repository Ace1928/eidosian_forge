import asyncio
import functools
import logging
import time
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Dict, Tuple, TypeVar, get_type_hints

# Type variable for the decorator
F = TypeVar("F", bound=Callable[..., Any])


def standard_decorator(
    retries: int = 3,
    delay: int = 2,
    cache_results: bool = False,
    log_performance: bool = True,
    validation_rules: Dict[str, Callable[[Any], bool]] = None,
) -> Callable[[F], F]:
    """
    A decorator that enhances a function with comprehensive features including logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching, and
    dynamic input validation and sanitization.

    Args:
        retries (int): Number of times to retry the function on failure.
        delay (int): Delay between retries in seconds.
        cache_results (bool): Whether to cache the function's return value.
        log_performance (bool): Whether to log the performance of the function.
        validation_rules (Dict[str, Callable[[Any], bool]]): Custom validation rules for function arguments.

    Returns:
        Callable[[F], F]: A wrapped function with enhanced capabilities.
    """
    if validation_rules is None:
        validation_rules = {}

    def decorator(func: F) -> F:
        cache: Dict[Tuple, Any] = {}
        arg_types = get_type_hints(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal cache
            validate_inputs(func, *args, **kwargs)
            key = (args, frozenset(kwargs.items()))
            if cache_results and key in cache:
                logging.debug(
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                return cache[key]

            attempts = 0
            while attempts < retries:
                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    if log_performance:
                        logging.debug(
                            f"{func.__name__} executed in {end_time - start_time:.2f}s"
                        )
                    if cache_results:
                        cache[key] = result
                    return result
                except Exception as e:
                    logging.error(
                        f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                    )
                    attempts += 1
                    await asyncio.sleep(delay)
            logging.debug(f"Final attempt for {func.__name__}")
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal cache
            validate_inputs(func, *args, **kwargs)
            key = (args, frozenset(kwargs.items()))
            if cache_results and key in cache:
                logging.debug(
                    f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                return cache[key]

            attempts = 0
            while attempts < retries:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    if log_performance:
                        logging.debug(
                            f"{func.__name__} executed in {end_time - start_time:.2f}s"
                        )
                    if cache_results:
                        cache[key] = result
                    return result
                except Exception as e:
                    logging.error(
                        f"{func.__name__} attempt {attempts + 1} failed with {e}, retrying..."
                    )
                    attempts += 1
                    time.sleep(delay)
            logging.debug(f"Final attempt for {func.__name__}")
            return func(*args, **kwargs)  # Final attempt without catching exceptions

        def validate_inputs(func: F, *args: Any, **kwargs: Any) -> None:
            bound_arguments = signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            for arg, value in bound_arguments.arguments.items():
                # Type checking
                if arg in arg_types and not isinstance(value, arg_types[arg]):
                    raise TypeError(f"Argument {arg} must be of type {arg_types[arg]}")
                # Custom validation rules
                if arg in validation_rules and not validation_rules[arg](value):
                    raise ValueError(
                        f"Validation failed for argument {arg} with value {value}"
                    )

        return async_wrapper if iscoroutinefunction(func) else sync_wrapper

    return decorator
