# Import necessary modules for the decorator functionality
from functools import wraps
from typing import Any, Callable
import logging


# Define the decorator to log function calls
def standard_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A universal/standard decorator for all Neuro Forge programs that systematically
    and consistently applies staticmethod or async method decoration dynamically.
    It also logs function entry and exit and handles error logging and raising.
    It will also implement type checking and any other kind of standardisation desired across
    the Neuro Forge program suite consistently and universally.
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

        if iscoroutinefunction(func):

            async def async_logger(*args, **kwargs):
                logging.debug(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                try:
                    result = await func(*args, **kwargs)
                    logging.debug(f"{func.__name__} returned {result}")
                    return result
                except Exception as e:
                    logging.error(f"{func.__name__} raised an exception: {e}")
                    raise

            return async_logger
        else:
            return staticmethod(logger)

    return wrapper  # Return the wrapper function
