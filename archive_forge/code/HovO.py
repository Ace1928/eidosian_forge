import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
    Dict,
    Any,
    Optional,
    TypeAlias,
    Callable,
    Awaitable,
    TypeVar,
    Coroutine,
    Union,
    Tuple,
    Type,
)
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import concurrent_log_handler
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])

# Module Header
"""
Module Name: indelogging.py
Description: This module provides a comprehensive and advanced logging setup for the INDEGO project development.
             It includes detailed configuration for various logging formats, handlers, and a custom asynchronous
             logging decorator to ensure non-blocking logging operations. It also ensures the logging configuration
             file exists or creates it with default settings.
Author: [Author Name]
Created Date: [Date]
Last Modified: [Date]
"""

# Type Aliases
DIR_NAME: TypeAlias = str
DIR_PATH: TypeAlias = pathlib.Path
FILE_NAME: TypeAlias = str
FILE_PATH: TypeAlias = pathlib.Path
LogFunction: TypeAlias = Callable[..., Awaitable[None]]
true: TypeAlias = bool  # For Correct JSON Formatting
false: TypeAlias = bool  # For Correct JSON Formatting
T = TypeVar("T")
ValidationRule = Callable[[Any], Awaitable[bool]]
ValidationRules = Dict[str, ValidationRule]

# Define the type for the decorator that can handle both coroutine and regular functions.
Decorator = Callable[
    [Callable[..., Union[T, Coroutine[Any, Any, T]]]],
    Callable[..., Coroutine[Any, Any, T]],
]
# Constants
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(process)d - %(thread)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
        "verbose": {
            "format": "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(processName)s - %(threadName)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
        "ascii_art": {
            "format": "#########################################################\\n# %(asctime)s - %(levelname)s - %(module)s\\n# Function: %(funcName)s - Line: %(lineno)d\\n# Process: %(process)d - Thread: %(thread)d\\n# Message: \\n# %(message)s\\n#########################################################",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "%",
            "validate": true,
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "ascii_art",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "errors_detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "verbose",
            "encoding": "utf-8",
            "delay": false,
        },
        "async_console": {
            "level": "DEBUG",
            "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "filename": "async_detailed.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
        "ascii_art_file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "ascii_art.log",
            "maxBytes": 52428800,
            "backupCount": 10,
            "formatter": "ascii_art",
            "encoding": "utf-8",
            "delay": false,
        },
    },
    "loggers": {
        "": {
            "handlers": [
                "console",
                "file",
                "errors",
                "async_console",
                "ascii_art_file",
            ],
            "level": "DEBUG",
            "propagate": true,
        },
        "async_logger": {
            "handlers": ["async_console"],
            "level": "DEBUG",
            "propagate": true,
        },
        "ascii_art_logger": {
            "handlers": ["ascii_art_file"],
            "level": "DEBUG",
            "propagate": true,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file", "errors", "async_console", "ascii_art_file"],
    },
}


class AsyncLogDecorator:
    def __init__(
        self,
        retries: int = 3,
        delay: int = 2,
        log_config: Dict = DEFAULT_LOGGING_CONFIG,
        validation_rules: Optional[Dict[str, Callable[[Any], bool]]] = None,
        retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        enable_performance_logging: bool = True,
        dynamic_retry_enabled: bool = True,
    ) -> None:
        self.retries = retries
        self.delay = delay
        self.log_config = log_config
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.enable_performance_logging = enable_performance_logging
        self.dynamic_retry_enabled = dynamic_retry_enabled

    def __call__(self, func: Callable) -> Callable:
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

    async def wrapper_logic(
        self, func: Callable, is_async: bool, *args, **kwargs
    ) -> Any:
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
        await validate_async_rules(func, *args, **kwargs)
        # key = self.cache_key_strategy(func, args, kwargs)
        # if self.cache_results and key in self.cache:
        #    return await self.cache_logic(key, func, *args, **kwargs)
        attempts = 0
        while attempts < self.retries:
            try:
                start_time = time.perf_counter()
                result = (
                    await func(*args, **kwargs) if is_async else func(*args, **kwargs)
                )
                end_time = time.perf_counter()
                if self.enable_performance_logging:
                    await log_performance(func, start_time, end_time)
                #        if self.cache_results:
                #           await self.cache_logic(
                #              key, func, *args, **kwargs
                #         )  # Cache the result
                return result
            except self.retry_exceptions as e:
                if self.dynamic_retry_enabled:
                    dynamic_retries, dynamic_delay = await dynamic_retry_strategy(
                        exception=e
                    )
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


# Dictionary mapping directory names to their respective paths. Optional typing allows for the possibility of uninitialized paths.
DIRECTORIES: Dict[DIR_NAME, Optional[DIR_PATH]] = {
    "ROOT": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development"),
    "LOGS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/logs"),
    "CONFIG": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config"
    ),
    "DATA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/data"),
    "MEDIA": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/media"),
    "SCRIPTS": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/scripts"
    ),
    "TEMPLATES": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/templates"
    ),
    "UTILS": pathlib.Path("/home/lloyd/EVIE/scripts/INDEGO_project_development/utils"),
}
# Dictionary mapping file names to their respective paths, ensuring that file paths are correctly typed and managed.
FILES: Dict[FILE_NAME, FILE_PATH] = {
    "DIRECTORIES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/directories.conf"
    ),
    "FILES_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/files.conf"
    ),
    "DATABASE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/database.conf"
    ),
    "API_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/api.conf"
    ),
    "CACHE_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/cache.conf"
    ),
    "LOGGING_CONF": pathlib.Path(
        "/home/lloyd/EVIE/scripts/INDEGO_project_development/config/logging.conf"
    ),
}


# Utility Functions
async def ensure_logging_config_exists(path: FILE_PATH) -> None:
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
                    DEFAULT_LOGGING_CONFIG, config_file, ensure_ascii=False, indent=4
                )
    except Exception as e:
        logging.error(f"Failed to ensure logging config exists: {e}", exc_info=True)


# Main Logging Configuration Setup
async def configure_logging() -> None:
    """
    Configures the logging system based on the logging configuration file. If the file does not exist,
    it ensures the file is created with the default configuration.
    """
    logging_conf_path: FILE_PATH = FILES["LOGGING_CONF"]
    await ensure_logging_config_exists(logging_conf_path)
    try:
        with logging_conf_path.open("r", encoding="utf-8") as config_file:
            logging_config = json.load(config_file)
            logging.config.dictConfig(logging_config)
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}", exc_info=True)


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


async def validate_async_rules(
    func: Callable[..., Awaitable[Any]],
    validation_rules: ValidationRules,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Asynchronously validates the inputs to the decorated function based on asynchronous validation rules. This method ensures that each argument
    passed to the function adheres to the predefined asynchronous validation rules, if any, enhancing the robustness and reliability of the function execution.

    This method meticulously inspects each argument provided to the asynchronous function, leveraging the power of Python's introspection capabilities
    to bind the provided arguments to the function's signature. This binding process allows for a detailed inspection and validation against the
    asynchronous validation rules defined within the class. If any argument fails to satisfy its corresponding asynchronous validation rule, an AsyncValidationException
    is raised, indicating that the argument's value is not acceptable for the function execution.

    Args:
        func (Callable[..., Awaitable[Any]]): The function being decorated, which may be an asynchronous function.
        validation_rules (ValidationRules): A dictionary mapping argument names to their respective asynchronous validation rules.
        *args (Any): Positional arguments passed to the function.
        **kwargs (Any): Keyword arguments passed to the function.

    Raises:
        AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule, indicating that the argument's value is not acceptable.
    """
    logging.debug(
        f"Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}"
    )

    # Binding the provided arguments to the function's signature enables detailed inspection and validation.
    bound_arguments = signature(func).bind(*args, **kwargs)
    bound_arguments.apply_defaults()  # Apply default values for any missing arguments to ensure completeness.

    # Iterating through each bound argument to validate against asynchronous rules.
    for arg, value in bound_arguments.arguments.items():
        if arg in validation_rules:
            validation_rule = validation_rules[arg]
            try:
                validation_result = (
                    await validation_rule(value)
                    if asyncio.iscoroutinefunction(validation_rule)
                    else validation_rule(value)
                )
                if not validation_result:
                    logging.error(
                        f"Validation failed for argument {arg} with value {value}"
                    )
                    raise AsyncValidationException(
                        arg,
                        value,
                        f"Validation failed for argument {arg} with value {value}",
                    )
            except AsyncValidationException as e:
                logging.exception("Async validation exception occurred", exc_info=e)
                raise
            except Exception as e:
                logging.exception(
                    f"Unexpected error during validation of argument {arg}", exc_info=e
                )
                raise AsyncValidationException(
                    arg, value, f"Unexpected validation error for argument {arg}"
                ) from e


async def log_performance(func: Callable, start_time: float, end_time: float) -> None:
    """
    Logs the performance of the decorated function, adjusting for decorator overhead.

    Args:
        func (F): The function that was executed.
        start_time (float): The start time of the function execution.
        end_time (float): The end time of the function execution.
    """
    overhead = 0.0001  # Example overhead value; adjust based on profiling
    adjusted_time = end_time - start_time - overhead
    logging.debug(f"{func.__name__} executed in {adjusted_time:.6f}s")


async def dynamic_retry_strategy(exception: BaseException) -> Tuple[int, int]:
    """
    Determines the retry strategy dynamically based on the exception type.

    Args:
        exception (Exception): The exception that triggered the retry logic.

    Returns:
        Tuple[int, int]: A tuple containing the number of retries and delay in seconds.
    """
    retries = 3
    delay = 2
    if isinstance(exception, TimeoutError):
        return (5, 1)  # More retries with a short delay for timeout errors.
    elif isinstance(exception, ConnectionError):
        return (3, 5)  # Fewer retries with a longer delay for connection errors.
    return (
        retries,
        delay,
    )  # Default strategy defined in the decorator attributes.


async def async_log_decorator(
    func: Callable[..., Union[T, Coroutine[Any, Any, T]]]
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    An advanced asynchronous decorator meticulously crafted to log the entry, exit, and exceptions of functions
    with unparalleled detail. This decorator is ingeniously designed to seamlessly handle both asynchronous and
    synchronous functions, executing them within an asynchronous event loop. It ensures exhaustive logging and
    process tracing by leveraging the sophisticated logging configuration established within the indelogging.py module.
    This decorator stands as a testament to the commitment to provide verbose and precise logging for every facet
    of function execution, thereby enhancing the observability and debuggability of the system.

    Args:
        func: The target function to decorate, which can be either synchronous or asynchronous in nature.

    Returns:
        A coroutine meticulously wrapping the original function, ensuring its execution is handled asynchronously
        with detailed logging at every step.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        logger = logging.getLogger(func.__module__)
        entry_message = f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}"
        exit_message = f"Exiting {func.__name__}"
        try:
            logger.info(entry_message)
            start_time = asyncio.get_event_loop().time()
            if asyncio.iscoroutinefunction(func):
                result: Union[T, Coroutine[Any, Any, T]] = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, functools.partial(func, *args, **kwargs)
                )
            end_time = asyncio.get_event_loop().time()
            logger.info(
                f"{exit_message} with result: {result} in {end_time - start_time:.2f}s"
            )
            if isinstance(result, Coroutine):
                return (
                    await result
                )  # Ensuring the coroutine result is awaited and returned correctly.
            return result
        except Exception as e:
            error_message = f"Exception in {func.__name__}: {str(e)}"
            logger.exception(error_message, exc_info=True)
            raise
        finally:
            if not logger.handlers:
                # Ensuring that the logger has handlers set up if it was not already configured.
                configure_logging_awaitable = configure_logging()
                await configure_logging_awaitable
            logger.info(exit_message)

    return wrapper


# Module Footer
"""
TODO:
- Investigate and integrate more advanced logging handlers and formatters for better log management.
- Explore the possibility of adding user-defined logging levels for more granular control over logging output.
- Implement a log rotation mechanism to manage log file sizes and prevent excessive disk space usage.
- Enhance the async_log_decorator to support configurable executors for better performance tuning.

Known Issues:
- None identified at this time.

Additional Functionalities:
- Future enhancements integration with indedatabase.py to store logs in a database for better log management.
- Future Enhancements: Integration with indecache.py for asynchronous smart caching of log messages.
- Future Enhancements: Integration with indeapi.py for logging API requests and responses for debugging purposes.
- Future Enhancements: Integration with indedecorators.py to contribute as a part of the decorator library for the INDEGO project.
- Future Enhancements: Integration with indeutils.py for additional utility functions and helper classes for logging operations.
"""
