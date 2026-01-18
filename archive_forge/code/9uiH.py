# Import necessary modules for the decorator functionality
from functools import wraps
from typing import Any, Callable
import logging


# Define the decorator to log function calls
def log_function_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that logs the entry, exit, arguments, and exceptions of the decorated function.

    This decorator is designed to enhance the debugging and monitoring capabilities by providing
    detailed insights into the function's execution flow. It logs the function name, arguments,
    return value, and any exceptions raised during the function call.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: A wrapper function that adds logging functionality to the original function.

    Example Usage:
        @log_function_call
        def example_function(param1, param2):
            return param1 + param2
    """

    # Use wraps to preserve the metadata of the original function
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate string representations of positional and keyword arguments
        args_repr = [
            repr(a) for a in args
        ]  # Convert positional arguments to their string representations
        kwargs_repr = [
            f"{k}={v!r}" for k, v in kwargs.items()
        ]  # Convert keyword arguments to string representations
        signature = ", ".join(
            args_repr + kwargs_repr
        )  # Combine all argument representations into a single string

        # Log the function call with its signature
        logging.debug(f"Calling {func.__name__} with arguments {signature}")

        try:
            # Attempt to execute the original function with the provided arguments
            result = func(*args, **kwargs)
            # Log the function's return value
            logging.debug(f"{func.__name__} returned {result!r}")
            return result  # Return the result of the function call
        except Exception as e:
            # Log any exception raised during the function call
            logging.error(
                f"{func.__name__} raised an exception {e.__class__.__name__}: {e}"
            )
            raise  # Reraise the exception for further handling

    return wrapper  # Return the wrapper function
