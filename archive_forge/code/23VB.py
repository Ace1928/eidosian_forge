__all__ = ["StandardDecorator", "import_from_path", "setup_logging"]
# Comprehensive import statements with detailed explanations for clarity and maintainability
import collections  # Provides support for ordered dictionaries, which are instrumental in the implementation of the caching mechanism, ensuring FIFO cache eviction logic.
import logging  # Facilitates comprehensive setup_logging capabilities, enabling detailed monitoring and debugging throughout the decorator's operation.
import asyncio  # Essential for the support of asynchronous operations, allowing the decorator to enhance both synchronous and asynchronous functions seamlessly.
import functools  # Offers utilities for working with higher-order functions and operations on callable objects, crucial for the decorator's wrapping mechanism.
import time  # Integral for the execution time measurement and the implementation of retry delays, providing accurate performance metrics and controlled operation retries.
import logging.handlers  # Extends the setup_logging module with additional handlers, enabling log management and storage in various formats and locations.
import threading  # Enables the creation of separate threads for log management, ensuring that setup_logging operations do not block the main application flow.
import importlib.util  # Facilitates dynamic module loading from file paths, allowing the decorator to import external modules and extend its functionality.
import types  # Provides support for creating new types dynamically, enhancing the decorator's type hinting capabilities and code clarity.
from inspect import (
    signature,  # Empowers the decorator with the ability to inspect callable objects, ensuring accurate argument binding and validation.
    BoundArguments,  # Facilitates the binding of arguments to their respective parameters, enabling dynamic input validation based on function signatures.
    Parameter,  # Represents the parameters in function signatures, used in conjunction with signature inspection to enforce type and value constraints.
    iscoroutinefunction,  # Determines if a callable is an asynchronous function, allowing the decorator to adapt its wrapping logic accordingly.
    iscoroutine,  # Identifies coroutine objects, enabling the decorator to handle asynchronous operations and results properly.
)

# These imports from the inspect module are pivotal for the decorator's ability to introspect and manipulate callable objects and their signatures.
from typing import (
    Any,  # Denotes an arbitrary type, allowing for flexible type hinting throughout the decorator's implementation.
    Callable,  # Signifies callable objects, providing a basis for generic function annotations and enhancing the decorator's versatility.
    Dict,  # Specifies a dictionary type, used extensively for type hinting in caching mechanisms and input validation rules.
    Tuple,  # Indicates a tuple type, utilized for function annotations involving multiple return types or fixed-size collections.
    TypeVar,  # Facilitates the creation of generic type variables, enabling type-safe function annotations and enhancing code readability.
    Optional,  # Represents optional types, allowing for the specification of parameters and return values that may or may not be present.
    Type,  # Denotes class types, used in specifying exception types for retry logic and enhancing the decorator's error handling capabilities.
    get_type_hints,  # Retrieves type hints from a callable, instrumental in implementing dynamic input validation based on type annotations.
    get_origin,  # Extracts the origin type from generics, aiding in the handling of complex type annotations and ensuring accurate type checking.
    get_args,  # Retrieves the arguments of generics, crucial for the detailed inspection of complex type annotations in validation logic.
    Union,  # Represents a union of types, used in function annotations to specify multiple allowable types for parameters and return values.
)  # These imports from the typing module are foundational for the decorator's type hinting and annotation capabilities, ensuring type safety and clarity.
import tracemalloc  # Activates memory usage tracking, enabling the identification of memory leaks and optimizing the decorator's memory footprint.
from functools import (
    wraps,
)  # Simplifies the creation of decorators by preserving function metadata, ensuring that wrapped functions retain their original attributes.
import inspect  # Provides tools for inspecting live objects, enabling the decorator to introspect functions and validate inputs based on their signatures.
import pickle  # Supports serialization and deserialization of Python objects, essential for caching results and managing the decorator's cache mechanism.
import os  # Facilitates interaction with the operating system, enabling the decorator to manage file paths and dynamically import modules.
import sys  # Provides access to Python runtime information, allowing the decorator to handle system-specific operations and configurations.

# Type variable F, bound to Callable, for generic function annotations
F = TypeVar("F", bound=Callable[..., Any])


def import_from_path(name: str, path: str) -> types.ModuleType:
    """
    Dynamically imports a module from a given file path.

    Args:
        name (str): The name of the module.
        path (str): The file path to the module.

    Returns:
        types.ModuleType: The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def setup_logging(log_level=logging.DEBUG):
    """
    Sets up setup_logging with both terminal and file handlers.
    Logs are managed in a separate thread to avoid blocking the main application flow.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Terminal handler
    terminal_handler = logging.StreamHandler()
    terminal_formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
    )
    terminal_handler.setFormatter(terminal_formatter)
    logger.addHandler(terminal_handler)
    # File handler with rotating logs
    file_handler = logging.handlers.RotatingFileHandler(
        "application.log", maxBytes=1048576, backupCount=5
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # Ensure setup_logging is done in a separate thread

    def log_thread():
        while True:
            pass  # Keep the thread alive

    logging_thread = threading.Thread(
        target=log_thread, name="loggingThread", daemon=True
    )
    logging_thread.start()


class StandardDecorator:
    """
    A class encapsulating a decorator that enhances functions with setup_logging, error handling,
    performance monitoring, automatic retrying on transient failures, optional result caching,
    and dynamic input validation and sanitization. Designed for use across the Neuro Forge INDEGO (EVIE) project.

    This decorator provides a robust framework for enhancing function execution with features like
    automatic retries on specified exceptions, performance setup_logging, input validation, and result caching
    with a thread-safe LRU cache strategy. It allows for granular control over setup_logging levels and
    includes support for complex validation rules, making it highly customizable and adaptable to various needs.

    Example usage:
    @StandardDecorator(retries=2, delay=1, log_level=setup_logging.INFO, cache_maxsize=100)
    def my_function(param1):
        # Function body

    Attributes:
    retries (int): The number of retry attempts for the decorated function upon failure.
    delay (int): The delay (in seconds) between retry attempts.
    cache_results (bool): Flag to enable or disable result caching for the decorated function.
    log_level (int): The setup_logging level to be used for setup_logging messages.
    validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): A dictionary mapping argument names to validation functions.
    retry_exceptions (Tuple[Type[BaseException], ...]): A tuple of exception types that should trigger a retry.
    cache_maxsize (int): The maximum size of the cache (number of items) when result caching is enabled.
    enable_performance_setup_logging (bool): Flag to enable or disable performance setup_logging.
    dynamic_retry_enabled (bool): Flag to enable or disable the dynamic retry strategy.
    cache_key_strategy (Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]]): Custom function for generating cache keys.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: int = 1,
        cache_results: bool = True,
        log_level: int = logging.DEBUG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = {},
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        cache_maxsize: int = 100,
        enable_performance_logging: bool = True,
        dynamic_retry_enabled: bool = True,
        cache_key_strategy: Callable[
            [Callable, Tuple[Any, ...], Dict[str, Any]], Tuple[Any, ...]
        ] = None,
        enable_caching: bool = True,
        enable_validation: bool = True,
    ):
        self.retries = retries
        logging.debug(f"Retries set to {retries}")
        self.delay = delay
        logging.debug(f"Delay set to {delay}")
        self.cache_results = (
            cache_results and enable_caching
        )  # Only enable caching if both flags are True
        logging.debug(f"Cache results: {self.cache_results}")
        self.validation_rules = validation_rules or {} if enable_validation else {}
        logging.debug(f"Validation rules: {self.validation_rules}")
        self.retry_exceptions = retry_exceptions
        logging.debug(f"Retry exceptions: {self.retry_exceptions}")
        self.cache_maxsize = cache_maxsize
        logging.debug(f"Cache maxsize: {self.cache_maxsize}")
        self.enable_performance_logging = enable_performance_logging
        logging.debug(
            f"Performance setup_logging enabled: {self.enable_performance_logging}"
        )
        self.dynamic_retry_enabled = dynamic_retry_enabled
        logging.debug(f"Dynamic retry strategy enabled: {self.dynamic_retry_enabled}")
        self.cache_key_strategy = (
            cache_key_strategy if cache_key_strategy else self.generate_cache_key
        )
        logging.debug(f"Cache key strategy: {self.cache_key_strategy}")
        self.enable_caching = enable_caching
        logging.debug(f"Result caching enabled: {self.enable_caching}")
        self.enable_validation = enable_validation
        logging.debug(f"Input validation enabled: {self.enable_validation}")
        self.cache = collections.OrderedDict()  # Initialize cache if caching is enabled
        logging.debug(f"Cache initialized: {self.cache}")

    async def cache_logic(self, key: Tuple[Any, ...], func: F, *args, **kwargs) -> Any:
        """
        Handles the caching logic for the decorated function, including cache hits and maintaining cache size.

        This method dynamically determines whether the passed function `func` is an asynchronous coroutine or a
        synchronous function and executes it accordingly. If the result of the function execution is not already
        cached, it computes the result, caches it, and returns it. If the cache has reached its maximum size, the
        oldest item in the cache is removed before adding the new item. If the result is already cached, it retrieves
        and returns the result directly from the cache.

        Args:
            key (Tuple[Any, ...]): The key under which the result is stored in the cache. This key is generated based
                                   on the function and its arguments to uniquely identify the function call.
            func (F): The function to be executed and potentially cached. This can be either an asynchronous coroutine
                      or a synchronous function.
            *args: Positional arguments for the function `func`.
            **kwargs: Keyword arguments for the function `func`.

        Returns:
            Any: The result of the function execution, either retrieved from the cache or newly computed.

        Raises:
            Exception: Propagates any exceptions raised during the execution of `func`.
        """
        # Generate the cache key using the provided strategy, which uniquely identifies the function call.
        key = self.cache_key_strategy(func, args, kwargs)
        logging.debug(f"Generated cache key: {key}")

        # Check if the result is already cached.
        if key in self.cache:
            logging.debug(
                f"Cache hit for {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            return self.cache[key]
        else:
            # Execute the function and cache the result if caching is enabled.
            if self.cache_results:
                # Evict the oldest item if the cache has reached its maximum size.
                if len(self.cache) >= self.cache_maxsize:
                    self.cache.popitem(last=False)  # Remove the oldest item
                    logging.debug(
                        f"Cache size exceeded by {len(self.cache) - self.cache_maxsize}. Evicting oldest item for {func.__name__}"
                    )

                # Determine if the function is an asynchronous coroutine or a synchronous function and execute it accordingly.
                if asyncio.iscoroutinefunction(func):
                    result = await func(
                        *args, **kwargs
                    )  # Await the result if `func` is a coroutine.
                else:
                    result = func(
                        *args, **kwargs
                    )  # Directly call `func` if it is a synchronous function.

                # Cache the result and return it.
                self.cache[key] = result
                logging.debug(f"Cached result {result} for {key}")
                return result
            else:
                # If caching is not enabled, directly execute the function without caching the result.
                if asyncio.iscoroutinefunction(func):
                    return await func(
                        *args, **kwargs
                    )  # Await the result if `func` is a coroutine.
                else:
                    return func(
                        *args, **kwargs
                    )  # Directly call `func` if it is a synchronous function.

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
        logging.debug(f"Invalidating cache entries: {to_invalidate}")
        for key in to_invalidate:
            del self.cache[key]
        logging.debug(f"Cache invalidated. New cache size: {len(self.cache)}")

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
        logging.debug(f"Generated cache key: {kwargs_key} for {func.__name__}")
        return (func.__name__, args, kwargs_key)

    def dynamic_retry_strategy(self, exception: Exception) -> Tuple[int, int]:
        """
        Determines the retry strategy dynamically based on the exception type.

        Args:
            exception (Exception): The exception that triggered the retry logic.

        Returns:
            Tuple[int, int]: A tuple containing the number of retries and delay in seconds.
        """
        logging.debug(
            f"Default retry strategy for error type {exception}. Retries: {self.retries}, Delay: {self.delay}"
        )
        if isinstance(exception, TimeoutError):
            logging.debug(
                f"Timeout error encountered for error type {exception}. Adjusting retry strategy to 5 retries with a 1 second delay."
            )
            return (5, 1)  # More retries with a short delay for timeout errors.
        elif isinstance(exception, ConnectionError):
            logging.debug(
                f"Connection error encountered for error type {exception}. Adjusting retry strategy to 3 retries with a 5 second delay."
            )
            return (3, 5)  # Fewer retries with a longer delay for connection errors.
        return (
            self.retries,
            self.delay,
        )  # Default strategy defined in the decorator attributes.

    def log_performance(self, func: F, start_time: float, end_time: float) -> None:
        """
        Logs the performance of the decorated function, adjusting for decorator overhead.

        Args:
            func (F): The function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
        logging.debug("Performance logging being called")
        if self.enable_performance_setup_logging:
            overhead = 0.0001  # Example overhead value; adjust based on profiling, make this dynamic and calculated based on actual overhead
            adjusted_time = end_time - start_time - overhead
            logging.debug(f"{func.__name__} executed in {adjusted_time:.6f}s")

    def __call__(self, func: F) -> F:
        """
        Makes the class instance callable, allowing it to be used as a decorator. It wraps the
        decorated function in a new function that performs argument validation before executing
        the original function.

        Args:
            func (F): The function to be decorated.

        Returns:
            F: The wrapped function, which includes argument validation logic.
        """

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            logging.debug(f"Calling async wrapper for {func.__name__} at {start_time}")
            if self.enable_validation:
                logging.debug(
                    f"Validating arguments asynchronously for {func.__name__} because validation enabled."
                )
                await self.validate_arguments(func, *args, **kwargs)
                logging.debug(
                    f"Asynchronous validation successful for {func.__name__} using arguments {args} and {kwargs}"
                )
            end_time = asyncio.get_event_loop().time()
            logging.debug(
                f"Validation of {func.__name__} completed at {end_time} taking total time of {end_time - start_time} seconds"
            )
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()  # Use time.time() for synchronous functions
            logging.debug(f"Calling sync wrapper for {func.__name__} at {start_time}")
            if self.enable_validation:
                logging.debug(
                    f"Validating arguments synchronously for {func.__name__} because validation enabled."
                )
                asyncio.run(self.validate_arguments(func, *args, **kwargs))
                logging.debug(
                    f"Synchronous validation successful for {func.__name__} using arguments {args} and {kwargs}"
                )
                end_time = time.time()  # Use time.time() for synchronous functions
            logging.debug(
                f"Validation of {func.__name__} completed at {end_time} taking total time of {end_time - start_time} seconds"
            )
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            logging.debug(f"Function {func.__name__} is asynchronous.")
            return async_wrapper
        else:
            logging.debug(f"Function {func.__name__} is synchronous.")
            return sync_wrapper

    async def _validate_func_signature(self, func, *args, **kwargs):
        """
        Validates the function's signature against provided arguments and types.
        """
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validating function signature for {func.__name__} at {start_time}"
        )
        sig = signature(func)
        logging.debug(f"Function signature: {sig}")
        bound_args = sig.bind(*args, **kwargs)
        logging.debug(f"Bound arguments: {bound_args}")
        bound_args.apply_defaults()
        logging.debug(f"Bound arguments with defaults applied: {bound_args}")

        type_hints = get_type_hints(func)
        logging.debug(f"Type hints: {type_hints}")
        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name)
            logging.debug(
                f"Validating argument '{name}' with value '{value}' against expected type '{expected_type}'"
            )
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(
                f"Argument '{name}' must be of type {expected_type}, got type {type(value)}"
            )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                """
                Asynchronous wrapper function that applies the decorator's logic to an asynchronous function.
                """
                start_time = asyncio.get_event_loop().time()
                result = await self.wrapper_logic(func, True, *args, **kwargs)
                end_time = asyncio.get_event_loop().time()
                logging.debug(
                    f"Validation of {func.__name__} completed at {end_time} taking total time of {end_time - start_time} seconds"
                )
                return result

            end_time = asyncio.get_event_loop().time()
            logging.debug(
                f"Validation of {func.__name__} completed at {end_time} taking total time of {end_time - start_time} seconds"
            )
            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                logging.debug(
                    f"Calling sync wrapper for {func.__name__} at {start_time}"
                )
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
                    end_time = asyncio.get_event_loop().time()
                    logging.debug(
                        f"Function {func.__name__} executed successfully in {end_time - start_time:.6f}s with result {future}"
                    )
                    return future
                else:
                    # No running event loop, safe to use asyncio.run
                    end_time = asyncio.get_event_loop().time()
                    logging.debug(
                        f"Function {func.__name__} executed successfully in {end_time - start_time:.6f}s with result {result}"
                    )
                return asyncio.run(self.wrapper_logic(func, False, *args, **kwargs))

            return sync_wrapper

    def solve_for(self, name: str, *args, **kwargs):
        """
        Dynamically executes a method based on its name, passing any provided arguments.
        """
        start_time = asyncio.get_event_loop().time()
        logging.debug(f"Solving for {method_name} at {start_time}")
        logging.debug(
            f"Solving for {method_name} with args: {args} and kwargs: {kwargs}"
        )
        method_name = f"do_{method_name}"
        logging.debug(f"Method name: {method_name}")
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            logging.debug(f"Method found: {method}")
            if callable(method):
                logging.debug(f"Method {method_name} is callable.")
                result = method(*args, **kwargs)
                end_time = asyncio.get_event_loop().time()
                logging.debug(
                    f"Validation of {name} completed at {end_time} taking total time of {end_time - start_time} seconds"
                )
                return result
            else:
                end_time = asyncio.get_event_loop().time()
                logging.debug(
                    f"Validation of {name} completed at {end_time} taking total time of {end_time - start_time} seconds"
                )
                logging.debug(f"Method {method_name} is not callable.")
                raise AttributeError(
                    f"Method '{method_name}' not found or is not callable."
                )
        else:
            end_time = asyncio.get_event_loop().time()
            logging.debug(
                f"Method {method_name} not found at {end_time} taking total time of {end_time - start_time} seconds"
            )
            raise AttributeError(
                f"Method '{method_name}' not found or is not callable."
            )

    async def wrapper_logic(self, func: F, is_async: bool, *args, **kwargs) -> Any:
        """
        Contains the core logic for retrying, caching, setup_logging, and now validating the execution of the decorated function.

        Args:
            func (F): The function to be executed.
            is_async (bool): Indicates whether the function is asynchronous.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution.
        """
        if is_async:
            start_time = asyncio.get_event_loop().time()
        else:
            start_time = time.time()  # Use time.time() for synchronous functions
        logging.debug(f"Executing wrapper logic for {func.__name__} at {start_time}")
        if self.enable_validation:
            logging.debug(f"Validation enabled for {func.__name__}")
            # Perform dynamic argument validation
            await self.dynamic_validate_arguments(func, *args, **kwargs)
            logging.debug(f"Validation successful for {func.__name__}")
        if self.cache_results:
            logging.debug(f"Cache results enabled for {func.__name__}")
            # Handle caching logic
            return await self.cache_logic(
                self.cache_key_strategy(func, args, kwargs), func, *args, **kwargs
            )
        logging.debug(f"Executing {func.__name__} at {start_time}")
        try:
            logging.debug(
                f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}"
            )
            if is_async:
                logging.debug(f"Function {func.__name__} is asynchronous.")
                result = await func(*args, **kwargs)
            else:
                logging.debug(f"Function {func.__name__} is synchronous.")
                result = func(*args, **kwargs)
        except self.retry_exceptions as e:
            logging.error(f"Error encountered: {e}. Retrying...")
            retry_strategy = self.dynamic_retry_strategy(e)
            logging.debug(f"Dynamic retry strategy: {retry_strategy}")
            for _ in range(retry_strategy[0]):
                logging.debug(
                    f"Retrying {func.__name__} with args: {args} and kwargs: {kwargs} using dynamic strategy {retry_strategy}"
                )
                try:
                    await asyncio.sleep(retry_strategy[1])
                    if is_async:
                        result = await func(*args, **kwargs)
                        logging.debug(
                            f"Function {func.__name__} executed successfully."
                        )
                        return result
                    else:
                        result = func(*args, **kwargs)
                        logging.debug(
                            f"Function {func.__name__} executed successfully."
                        )
                        return result
                except self.retry_exceptions as retry_e:
                    logging.error(f"Retry error encountered: {retry_e}")
            raise
        finally:
            if is_async:
                end_time = asyncio.get_event_loop().time()
            else:
                end_time = time.time()  # Use time.time() for synchronous functions
            logging.debug(f"Execution of {func.__name__} completed at {end_time}")
            self.log_performance(func, start_time, end_time)
            logging.debug(
                f"Function {func.__name__} executed successfully in {end_time - start_time:.6f}s with result {result}"
            )
        return result

    def _get_arg_position(self, func: F, arg_name: str) -> int:
        """
        Determines the position of an argument in the function's signature.

        Args:
            func (F): The function being inspected.
            arg_name (str): The name of the argument whose position is sought.

        Returns:
            int: The position of the argument in the function's signature.
        """
        start_time = time.time()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} at {start_time}"
        )
        result = list(signature(func).parameters).index(arg_name)
        logging.debug(f"Argument position for {arg_name} in {func.__name__}: {result}")
        end_time = time.time()
        logging.debug(
            f"Getting arg position for {arg_name} in {func.__name__} completed at {end_time}"
        )
        return result

    async def validate_arguments(self, func: F, *args, **kwargs) -> None:
        """
        Validates the arguments passed to a function against expected type hints and custom validation rules.
        Adjusts for whether the function is a bound method (instance or class method) or a regular function
        or static method, and applies argument validation accordingly.
        """
        start_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validating arguments for {func.__name__} at {start_time} with args: {args} and kwargs: {kwargs}"
        )

        # Adjust args for bound methods (instance or class methods)
        if inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        ):
            # For bound methods, the first argument ('self' or 'cls') should not be included in validation
            args = args[1:]

        # Attempt to bind args and kwargs to the function's signature
        try:
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
        except TypeError as e:
            logging.error(f"Error binding arguments for {func.__name__}: {e}")
            raise

        bound_args.apply_defaults()
        type_hints = get_type_hints(func)

        for name, value in bound_args.arguments.items():
            expected_type = type_hints.get(name)
            if (
                expected_type is Any
                or getattr(expected_type, "__origin__", None) is Any
            ):
                logging.debug(
                    f"Skipping type check for '{name}' as the expected type is typing.Any"
                )
                continue

            if expected_type and not isinstance(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type '{expected_type.__name__}', got '{type(value).__name__}'"
                )

            validation_rule = self.validation_rules.get(name)
            if validation_rule:
                valid = (
                    await validation_rule(value)
                    if asyncio.iscoroutinefunction(validation_rule)
                    else validation_rule(value)
                )
                if not valid:
                    raise ValueError(
                        f"Validation failed for argument '{name}' with value '{value}'"
                    )

        end_time = asyncio.get_event_loop().time()
        logging.debug(
            f"Validation completed at {end_time} taking total time of {end_time - start_time} seconds"
        )
