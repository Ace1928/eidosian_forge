import logging
import logging.config
import logging.handlers
import sys
import time
import asyncio
import aiofiles
from typing import (
import pathlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
import functools
from functools import wraps
import tracemalloc
import inspect
from inspect import signature, Parameter
from IndegoValidation import AsyncValidationException, ValidationRules
class UniversalDecorator:
    """
    A highly sophisticated decorator designed to enhance the functionality of both synchronous and asynchronous functions.
    It incorporates advanced features such as automatic retry with exponential backoff, input validation based on custom rules,
    performance logging, and dynamic adjustment of retry strategies based on exceptions encountered during function execution.
    This decorator is meticulously crafted to ensure maximum flexibility, robustness, and efficiency in handling a wide range
    of use cases, making it an indispensable tool for any Python project.

    Attributes:
        retries (int): The maximum number of retries for the decorated function.
        delay (int): The initial delay between retries, which may be dynamically adjusted.
        log_config (Dict): The logging configuration to be used for logging within the decorator.
        validation_rules (Optional[Dict[str, Callable[[Any], bool]]]): Custom validation rules for function arguments.
        retry_exceptions (Tuple[Type[BaseException], ...]): Exceptions that trigger a retry.
        enable_performance_logging (bool): Flag to enable or disable performance logging.
        dynamic_retry_enabled (bool): Flag to enable or disable dynamic retry strategies.
    """

    def __init__(self, retries: int=3, delay: int=2, log_config: Dict[str, Any]=DEFAULT_LOGGING_CONFIG, validation_rules: Optional[ValidationRules]=None, retry_exceptions: Tuple[Type[BaseException], ...]=(Exception,), enable_performance_logging: bool=True, dynamic_retry_enabled: bool=True) -> None:
        self.retries = retries
        self.delay = delay
        self.log_config = log_config
        self.validation_rules = validation_rules or {}
        self.retry_exceptions = retry_exceptions
        self.enable_performance_logging = enable_performance_logging
        self.dynamic_retry_enabled = dynamic_retry_enabled

    def __call__(self, func: Callable[..., Union[Awaitable[Any], Any]]) -> Callable[..., Union[Awaitable[Any], Any]]:
        """
        Transforms the UniversalDecorator into a callable object, allowing it to be used as a decorator. This method
        dynamically determines whether the decorated function is synchronous or asynchronous and applies the appropriate
        wrapper to enhance its functionality with retries, validation, performance logging, and dynamic retry strategies.

        Args:
            func (Callable[..., Union[Awaitable[Any], Any]]): The function to be decorated.

        Returns:
            Callable[..., Union[Awaitable[Any], Any]]: The decorated function with enhanced functionality.
        """
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                An asynchronous wrapper function that provides the core functionality of the UniversalDecorator for asynchronous functions.
                It incorporates logic for validation, retrying with exponential backoff, performance logging, and dynamic retry strategy adjustment.

                Args:
                    *args: Positional arguments for the decorated function.
                    **kwargs: Keyword arguments for the decorated function.

                Returns:
                    Any: The result of the decorated function execution.
                """
                if self.validation_rules:
                    await self.validate_rules_async(func, *args, **kwargs)
                attempts = 0
                delay = self.delay
                while attempts < self.retries:
                    try:
                        start_time = time.perf_counter()
                        result = await func(*args, **kwargs)
                        end_time = time.perf_counter()
                        if self.enable_performance_logging:
                            await self.log_performance_async(func.__name__, start_time, end_time)
                        return result
                    except self.retry_exceptions as e:
                        attempts += 1
                        logging.error(f'{func.__name__} attempt {attempts} failed with {e}, retrying after {delay} seconds...')
                        if self.dynamic_retry_enabled:
                            delay = await self.dynamic_retry_strategy_async(e, attempts)
                        await asyncio.sleep(delay)
                logging.debug(f'Final attempt for {func.__name__}')
                return await func(*args, **kwargs)
            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                A synchronous wrapper function that directly executes the decorated synchronous function without event loop manipulation.
                This function retains core functionalities such as validation, retrying with exponential backoff, performance logging, and dynamic retry strategy adjustment,
                while simplifying the execution path for synchronous functions.

                Args:
                    *args: Positional arguments for the decorated function.
                    **kwargs: Keyword arguments for the decorated function.

                Returns:
                    Any: The result of the decorated function execution.
                """
                attempts = 0
                delay = self.delay
                while attempts < self.retries:
                    try:
                        start_time = time.perf_counter()
                        if self.validation_rules:
                            self.validate_rules_sync(func, *args, **kwargs)
                        result = func(*args, **kwargs)
                        end_time = time.perf_counter()
                        if self.enable_performance_logging:
                            self.log_performance_sync(func.__name__, start_time, end_time)
                        return result
                    except self.retry_exceptions as e:
                        attempts += 1
                        logging.error(f'{func.__name__} attempt {attempts} failed with {e}, retrying after {delay} seconds...')
                        if self.dynamic_retry_enabled:
                            delay = self.dynamic_retry_strategy_sync(e, attempts)
                        time.sleep(delay)
                logging.debug(f'Final attempt for {func.__name__}')
                return func(*args, **kwargs)
            return sync_wrapper

    async def validate_rules_async(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
        """
        Asynchronously validates the inputs to the decorated function based on the provided asynchronous validation rules.
        This method ensures that each argument passed to the function adheres to the predefined rules, enhancing the robustness
        and reliability of the function execution.

        Args:
            func (Callable[..., Awaitable[Any]]): The function being decorated.
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Raises:
            AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule.
        """
        logging.debug(f'Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}')
        if not hasattr(self, '_bound_arguments_checked'):
            bound_arguments = signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            self._bound_arguments_checked = True
            for arg, value in bound_arguments.arguments.items():
                if arg in self.validation_rules:
                    validation_rule = self.validation_rules[arg]
                    is_valid = await validation_rule(value) if asyncio.iscoroutinefunction(validation_rule) else validation_rule(value)
                    if not is_valid:
                        raise AsyncValidationException(arg, value, f"Validation failed for argument '{arg}' with value '{value}'")

    async def log_performance_async(self, func_name: str, start_time: float, end_time: float) -> None:
        """
        Logs the performance of the decorated function, including the time taken for execution.

        Args:
            func_name (str): The name of the function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
        adjusted_time = end_time - start_time
        logging.debug(f'{func_name} executed in {adjusted_time:.6f}s')

    async def dynamic_retry_strategy_async(self, exception: BaseException, attempt: int) -> int:
        """
        Dynamically determines the retry delay based on the exception type and the number of attempts already made.
        This method allows for a more adaptive and responsive retry mechanism, potentially increasing the chances of success in subsequent attempts.

        Args:
            exception (BaseException): The exception that triggered the retry logic.
            attempt (int): The current retry attempt number.

        Returns:
            int: The delay in seconds before the next retry attempt.
        """
        if isinstance(exception, TimeoutError):
            return min(5, 2 ** attempt)
        elif isinstance(exception, ConnectionError):
            return min(10, 2 * attempt)
        return self.delay

    def log_performance_sync(self, func_name: str, start_time: float, end_time: float) -> None:
        """
        Logs the performance of the decorated function, including the time taken for execution.

        Args:
            func_name (str): The name of the function that was executed.
            start_time (float): The start time of the function execution.
            end_time (float): The end time of the function execution.
        """
        adjusted_time = end_time - start_time
        logging.debug(f'{func_name} executed in {adjusted_time:.6f}s')

    def dynamic_retry_strategy_sync(self, exception: BaseException, attempt: int) -> int:
        """
        Dynamically determines the retry delay based on the exception type and the number of attempts already made.
        This method allows for a more adaptive and responsive retry mechanism, potentially increasing the chances of success in subsequent attempts.

        Args:
            exception (BaseException): The exception that triggered the retry logic.
            attempt (int): The current retry attempt number.

        Returns:
            int: The delay in seconds before the next retry attempt.
        """
        if isinstance(exception, TimeoutError):
            return min(5, 2 ** attempt)
        elif isinstance(exception, ConnectionError):
            return min(10, 2 * attempt)
        return self.delay

    def validate_rules_sync(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
        """
        Asynchronously validates the inputs to the decorated function based on the provided asynchronous validation rules.
        This method ensures that each argument passed to the function adheres to the predefined rules, enhancing the robustness
        and reliability of the function execution.

        Args:
            func (Callable[..., Awaitable[Any]]): The function being decorated.
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Raises:
            AsyncValidationException: If any argument fails to satisfy its corresponding asynchronous validation rule.
        """
        logging.debug(f'Validating async rules for function {func.__name__} with args {args} and kwargs {kwargs}')
        if not hasattr(self, '_bound_arguments_checked'):
            bound_arguments = signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            self._bound_arguments_checked = True
            for arg, value in bound_arguments.arguments.items():
                if arg in self.validation_rules:
                    validation_rule = self.validation_rules[arg]
                    is_valid = asyncio.run(validation_rule(value)) if asyncio.iscoroutinefunction(validation_rule) else validation_rule(value)
                    if not is_valid:
                        raise AsyncValidationException(arg, value, f"Validation failed for argument '{arg}' with value '{value}'")