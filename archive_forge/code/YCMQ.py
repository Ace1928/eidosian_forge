"""
    CRITICALLY IMPORTANT DETAILS:
    # Floating Point Precision

        # 32-bit floating point numbers have a precision of 23 bits.
            Decimal Precision≈Binary_Precision×log10(2)
            Substituting the binary precision of 23 bits for a 32 bit float:
            Decimal_Precision≈23×log10(2)≈6.92
            Thus, a precision of 6 bits is chosen for the fractional part of the number as the 7th bit is unreliable.

        # 64-bit floating point numbers have a precision of 52 bits.
            Decimal Precision≈Binary_Precision×log10(2)
            Substituting the binary precision of 52 bits for a 64 bit float:
            Decimal Precision≈53×log10(2)≈53×0.3010≈15.953
            Thus, a precision of 15 bits is chosen for the fractional part of the number as the 16th bit is unreliable.

    #Scientific Notation
        # 32-bit floating point numbers have a range of approximately 10^-38 to 10^38.
        Single Precision (32-bit)
            Format: 1 bit for sign, 8 bits for exponent, and 23 bits for the fraction (mantissa).
            Bias: 127
            Exponent Range: -126 to +127 (after bias adjustment)
            Precision: Approximately 7 decimal places.
            In single precision, the number is represented as:
            (−1)sign×1.fraction×2exponent+−127 
            where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 7 decimal places comes from the 23 bits allocated for the fraction part.

        # 64-bit floating point numbers have a range of approximately 10^-308 to 10^308.
        Double Precision (64-bit)
            Format: 1 bit for sign, 11 bits for exponent, and 52 bits for the fraction (mantissa).
            Bias: 1023
            Exponent Range: -1022 to +1023 (after bias adjustment)
            Precision: Approximately 16 decimal places.
            In double precision, the number is represented as:
            (−1)sign×1.fraction×2exponent+−1023
            where the "1." before the fraction part is the implicit leading bit (normalized form). The precision of about 16 decimal places comes from the 52 bits allocated for the fraction part.

    # Real World Degrees of Freedom
            - 1. X dimension # Length corresponds to the x-axis.
            - 2. Y dimension # Width corresponds to the y-axis.
            - 3. Z dimension # Height corresponds to the z-axis.
            - 4. T dimension # Time corresponds to the t-axis.
            - 5. X Rotation # Roll corresponds to the orientation in radians around the x-axis.
            - 6. Y Rotation # Pitch corresponds to the orientation in radians around the y-axis.
            - 7. Z Rotation # Yaw corresponds to the orientation in radians around the z-axis.
            - 8. T Rotation # Time corresponds to the orientation in radians around the t-axis.
            - 9. X Scale Factor # X Scale corresponds to the scaling factor along the x-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
            - 10. Y Scale Factor # Y Scale corresponds to the scaling factor along the y-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
            - 11. Z Scale Factor # Z Scale corresponds to the scaling factor along the z-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
            - 12. T Scale Factor # T Scale corresponds to the scaling factor along the t-axis. From 0 to -1 for contraction, 0 - 1 for expansion.
            - 13. X Curvature # X Skew corresponds to the curvature along the x-axis. From 0 to -1 for concave, 0 - 1 for convex.
            - 14. Y Curvature # Y Skew corresponds to the curvature along the y-axis. From 0 to -1 for concave, 0 - 1 for convex.
            - 15. Z Curvature # Z Skew corresponds to the curvature along the z-axis. From 0 to -1 for concave, 0 - 1 for convex.
            - 16. T Curvature # T Skew corresponds to the curvature along the t-axis. From 0 to -1 for concave, 0 - 1 for convex.

    # Bit Matrix Representation of Floating Point Numbers
    # Arbitrary Precision - Precision Affects Matrix Size
        # 2 versions of system.
        # Each version producing a different matrix representation and requiring different operations.
        # Kind of akin to chirality in chemistry and biology and physics.
            1. 0 for negative, 1 for positive; 0 for absent, 1 for present. (Left Handed)
            2. 1 for negative, 0 for positive; 1 for absent, 0 for present. (Right Handed)
        Overall Representation: {
            ([Version(1 value)], # Maximum Number of Values = 1
            [Channel(9 values)], # Maximum Number of Values = 9
            [Count((16xChannel(for each value not none)) values)], # Maximum Number of Values = 144 
            [Sign((9xCount(for each not none) values)],  # Maximum Number of Values = 9 x 144 = 1296
            [Figures(min length of values in count, default 6)],  # Maximum Number of Values = 1
            [Value(For each sig figuresxnumber Channelsxnumber of Count and each Value 0 - 9 recorded as binary mapping)],  # Maximum Number of Values = 6 x 9 x 144 x 10 = 77760
            [Uncertainty( for each of the channels in each of the counts based on the last figure present. For Version 1, 0 for negative, 1 for positive. For Version 2, 1 for negative, 0 for positive.)]  # Maximum Number of Values = 1296 (one for each of the possible values that make up all possible channels and all possible counts)
        }

        # Version: 1 for left handed, 2 for right handed.
        # Channel: For all parts of the value from 1 real up to 8 (octonion) imaginary parts. 1 for present, 0 for absent.
        # Count: The number of components that make up the matrix (number of vectors) maximum of 16, for each channel, corresponding to comprehensive real world degrees of freedom. 
        # Sign: For version 1(L): 0 for negative, 1 for positive. For version 2(R): 1 for negative, 0 for positive.
        # Figures: The count for the minimum length of sig figures for the values in count OR a default of 6.
        # Value: The value of the sig figs in count.
        # Uncertainty: The uncertainty of the value of the sig figs in count.
        # Total Bits:
            Single Vector Single Dimension negative or positive:
            Value to Convert: -1.23456
                Version: 1
                Channels: 1 (only 1 real)
                Count: 1 (only 1 vector)
                Sign: 0 (only 1 sign and 1 channel and for version 1 0 corresponds to negative)
                Figures: 6 (6 significant figures in value) binary mapping for 6: 0110
                Value (binary mappy matrix for each sig fig): [(0,0,0,0), (1,0,0,1), (0,0,1,1), (0,1,0,0), (1,0,1,0), (1,1,0,0)]   
                Uncertainty: Last digit > 5 and Version = 1 so Uncertainty = 1
            Final Matrix: [
                ([(1)],[(1,0,0,0,0,0,0,0,0)],[(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)],[(0)],[(0110)],[(0,0,0,0),(1,0,0,1),(0,0,1,1),(0,1,0,0),(1,0,1,0),(1,1,0,0)],[(1)])
            ]
"""

import asyncio
import logging
import logging.config
from typing import List, Tuple, Optional, Union, Any
import functools
from functools import wraps
import cProfile
import pstats
import io
import tracemalloc
import signal
import sys
from contextlib import asynccontextmanager, contextmanager
from memory_profiler import profile
import cachetools.func
from cachetools import TTLCache
import aiofiles
import aiohttp
from aiohttp import web
import json
from datetime import datetime

# Configure logging with maximum verbosity and comprehensive profiling and tracing
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(process)d - %(thread)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "detailed",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix_detailed.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "formatter": "detailed",
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix_errors_detailed.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 20,
            "formatter": "detailed",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "errors"],
            "level": "DEBUG",
            "propagate": True,
        },
    },
}
logging.config.dictConfig(log_config)

# Initialize profiling and tracing with maximum verbosity
pr = cProfile.Profile()
pr.enable()
tracemalloc.start()

# Initialize cache with comprehensive logging for cache operations
cache = TTLCache(maxsize=2048, ttl=7200)  # Increased cache size and TTL

# Initialize web app with detailed logging for each request and response
app = web.Application(middlewares=[web.normalize_path_middleware()])


# Signal handler for graceful termination with detailed profiling and memory usage logging
def signal_handler(sig, frame):
    logging.info(
        "Received signal to terminate. Dumping detailed logs, profiles, and memory usage."
    )
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    logging.info(s.getvalue())
    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
    tracemalloc.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Async context manager for detailed profiling of code blocks
@asynccontextmanager
async def profiled():
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()
        logging.info("Detailed profiling info:\n" + s.getvalue())


# Decorator for wrapping synchronous functions to be run asynchronously with detailed profiling
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        async with profiled():
            result = await loop.run_in_executor(executor, pfunc)
            logging.debug(
                f"Function {func.__name__} executed asynchronously with result: {result}"
            )
            return result

    return run


# Decorator for caching results of expensive functions with detailed logging
def cache_result(key: str, maxsize: int = 256, ttl: int = 1200, typed: bool = True):
    def decorator(func):
        cache = TTLCache(maxsize=maxsize, ttl=ttl, typed=typed)

        @cachetools.func.ttl_cache(cache=cache)
        async def cached_func(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await cached_func(*args, **kwargs)
                logging.debug(
                    f"Cached result for {func.__name__} with key {key}: {result}"
                )
            except Exception as e:
                logging.exception(
                    f"Error in cached function {func.__name__} with args {args} and kwargs {kwargs}: {e}"
                )
                raise e
            return result

        return wrapper

    return decorator


@async_wrap
@cache_result(key="input_optional", maxsize=2048, ttl=3600)
async def input_optional(prompt: str) -> Optional[str]:
    """
    Prompt for optional input, return None if empty, with detailed logging.

    Args:
        prompt (str): The prompt message to display to the user.

    Returns:
        Optional[str]: The user's response or None if empty.
    """
    logging.debug(f"Prompting user with: {prompt}")
    response = input(prompt + " (leave blank if not applicable): ").strip()
    logging.debug(f"User response: {response}")
    return response if response else None


@async_wrap
@cache_result(key="convert_to_binary", maxsize=256)
async def convert_to_binary(value: int, length: int = 4) -> str:
    """
    Convert a decimal value to binary with fixed length.

    Args:
        value (int): The decimal value to convert.
        length (int): The fixed length of the binary representation.

    Returns:
        str: The binary representation of the value.
    """
    logging.debug(f"Converting {value} to binary with length {length}")
    try:
        binary = format(int(value), f"0{length}b")
    except (ValueError, TypeError) as e:
        logging.exception(f"Error converting {value} to binary: {e}")
        raise e
    logging.debug(f"Binary result: {binary}")
    return binary


@async_wrap
@cache_result(key="validate_and_parse_input", maxsize=512)
async def validate_and_parse_input(
    prompt: str, expected_type: Any = int, delimiter: str = ",", optional: bool = False
) -> List:
    """
    Prompt for input, validate, parse, and return a list of values of the expected type.

    Args:
        prompt (str): The prompt message for input.
        expected_type (Any): The expected type of each input value.
        delimiter (str): The delimiter used to separate input values.
        optional (bool): Whether the input is optional.

    Returns:
        List: A list of values of the expected type.
    """
    while True:
        logging.debug(f"Validating and parsing input for prompt: {prompt}")
        response = await input_optional(prompt)
        if optional and response is None:
            logging.debug("Optional input not provided")
            return []
        elif response is None:
            logging.debug("No input provided")
            return []
        try:
            values = [expected_type(item.strip()) for item in response.split(delimiter)]
            logging.debug(f"Parsed values: {values}")
            return values
        except (ValueError, TypeError) as e:
            logging.exception(
                f"Invalid input. Please ensure your input matches the expected format. Error: {e}"
            )


@async_wrap
@cache_result(key="construct_bit_matrix", maxsize=128)
async def construct_bit_matrix(
    version: int,
    channels: List[int],
    counts: List[int],
    signs: List[int],
    figures: int,
    values: List[int],
    uncertainties: List[int],
) -> List:
    """
    Construct the bit matrix representation.

    Args:
        version (int): Version of the matrix.
        channels (List[int]): List of channels.
        counts (List[int]): List of counts.
        signs (List[int]): List of signs.
        figures (int): Number of significant figures.
        values (List[int]): List of values.
        uncertainties (List[int]): List of uncertainties.

    Returns:
        List: The constructed bit matrix.
    """
    logging.debug("Constructing bit matrix")
    try:
        version_matrix = [version]
        channel_matrix = channels
        count_matrix = [len(counts)] + counts
        sign_matrix = signs
        figures_matrix = [figures]
        value_matrix = [await convert_to_binary(val) for val in values]
        uncertainty_matrix = uncertainties
        bit_matrix = [
            version_matrix,
            channel_matrix,
            count_matrix,
            sign_matrix,
            figures_matrix,
            value_matrix,
            uncertainty_matrix,
        ]
    except Exception as e:
        logging.exception(f"Error constructing bit matrix: {e}")
        raise e
    logging.debug(f"Constructed bit matrix: {bit_matrix}")
    return bit_matrix


@async_wrap
@cache_result(key="construct_matrix_handler", maxsize=128)
async def construct_matrix_handler(request):
    """
    Asynchronously handles the request to construct a bit matrix.

    This function parses the request JSON for parameters, validates and processes them,
    then constructs and returns the bit matrix via a JSON response.

    Args:
        request: The request object from the web server.

    Returns:
        A web response object containing the bit matrix or an error message.
    """
    try:
        # Parse and validate request data
        data = await request.json()
        version = data.get("version", 1)
        channels = data.get("channels", [1, 0, 0, 0, 0, 0, 0, 0, 0])
        counts = data.get("counts", [0])
        signs = data.get("signs", [1])
        figures = data.get("figures", 6)
        values = data.get("values", [0])
        uncertainties = data.get("uncertainties", [0])

        # Construct the bit matrix using the validated data
        bit_matrix = await construct_bit_matrix(
            version, channels, counts, signs, figures, values, uncertainties
        )
        return web.json_response({"bit_matrix": bit_matrix})
    except Exception as e:
        # Log and respond with an error if any occurs during processing
        logging.exception(f"Error in construct matrix handler: {e}")
        return web.json_response({"error": str(e)}, status=400)


@async_wrap
@cache_result(key="prompt_for_values", maxsize=64)
async def prompt_for_values() -> (
    Tuple[int, List[int], List[int], List[int], int, List[int], List[int]]
):
    """Prompt user for input and return a tuple of all components, with logic-based prompting."""
    logging.debug("Prompting for values")
    try:
        version = (
            await validate_and_parse_input(
                "Enter version (1 for Left Handed, 2 for Right Handed): ", optional=True
            )
        )[0] or 1
        channels = await validate_and_parse_input(
            "Enter channels (comma-separated, 1 for present, 0 for absent): ",
            optional=True,
        ) or [1, 0, 0, 0, 0, 0, 0, 0, 0]
        counts = await validate_and_parse_input(
            "Enter counts for each channel (comma-separated): ", optional=True
        ) or [0]
        signs = await validate_and_parse_input(
            "Enter signs for each count (comma-separated, 0 for negative, 1 for positive): ",
            optional=True,
        ) or [1]
        figures = (
            await validate_and_parse_input(
                "Enter minimum length of significant figures (default 6): ",
                optional=True,
            )
        )[0] or 6
        values = await validate_and_parse_input(
            "Enter values for significant figures (comma-separated, 0-9): ",
            optional=True,
        ) or [0]
        uncertainties = await validate_and_parse_input(
            "Enter uncertainties for each channel in each count (comma-separated, 0 for negative, 1 for positive for Version 1; reverse for Version 2): ",
            optional=True,
        ) or [0]
        logging.debug(
            f"Prompted values: version={version}, channels={channels}, counts={counts}, signs={signs}, figures={figures}, values={values}, uncertainties={uncertainties}"
        )
        return version, channels, counts, signs, figures, values, uncertainties
    except Exception as e:
        logging.exception(f"Error prompting for values: {e}")
        raise e


async def main():
    """
    Main function to drive the program.

    Sets up the web server routes and handles the main logic for constructing the bit matrix.
    """
    # Setup web server routes
    app.add_routes([web.post("/construct_matrix", construct_matrix_handler)])
    try:
        logging.info("Starting main function")
        # Await the coroutine for prompting values and ensure correct unpacking
        result = await prompt_for_values()
        if result is not None:
            version, channels, counts, signs, figures, values, uncertainties = result
            # Construct the bit matrix with the obtained values
            bit_matrix = await construct_bit_matrix(
                version, channels, counts, signs, figures, values, uncertainties
            )
            print(f"Constructed Bit Matrix Representation: {bit_matrix}")
        else:
            logging.error("No result from prompt_for_values, unable to proceed.")
    except Exception as e:
        # Handle any exceptions that occur during the main function execution
        logging.exception(f"Error in main function: {e}")
        raise e


if __name__ == "__main__":
    # Initialize and run the event loop for the asynchronous main function
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    # Start the web application
    web.run_app(app)
