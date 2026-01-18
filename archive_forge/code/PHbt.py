import asyncio
import logging
from typing import List, Tuple, Optional, Union, Any
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

# Configure logging with maximum verbosity
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s - %(levelname)s - %(module)s - %(process)d - %(thread)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "verbose",
        },
        "errors": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "bit_matrix_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "verbose",
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

# Initialize profiling
pr = cProfile.Profile()
pr.enable()

# Initialize memory tracing
tracemalloc.start()

# Initialize cache
cache = TTLCache(maxsize=1024, ttl=3600)

# Initialize web app
app = web.Application()


# Initialize signal handler for graceful termination
def signal_handler(sig, frame):
    logging.info("Received signal to terminate. Dumping logs and profiles.")
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


# Async context manager for profiling code blocks
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
        logging.info(s.getvalue())


# Decorator for wrapping synchronous functions to be run asynchronously
def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = profile(func)
        async with profiled():
            return await loop.run_in_executor(executor, pfunc, *args, **kwargs)

    return run


# Decorator for caching results of expensive functions
def cache_result(key: str, maxsize: int = 128, ttl: int = 600, typed: bool = False):
    def decorator(func):
        @cachetools.func.ttl_cache(maxsize=maxsize, ttl=ttl, typed=typed)
        async def cached_func(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await cached_func(*args, **kwargs)
            except Exception as e:
                logging.exception(
                    f"Error in {func.__name__} with args {args} and kwargs {kwargs}"
                )
                raise e

            async with aiofiles.open("bit_matrix_cache.json", mode="r") as f:
                data = json.loads(await f.read())
            data[key] = result
            async with aiofiles.open("bit_matrix_cache.json", mode="w") as f:
                await f.write(json.dumps(data, default=str))
            return result

        return wrapper

    return decorator


@async_wrap
@cache_result(key="input_optional", maxsize=1024)
async def input_optional(prompt: str) -> Optional[str]:
    """Prompt for optional input, return None if empty."""
    logging.debug(f"Prompting user with: {prompt}")
    response = input(prompt + " (leave blank if not applicable): ").strip()
    logging.debug(f"User response: {response}")
    return response if response else None


@async_wrap
@cache_result(key="convert_to_binary", maxsize=256)
async def convert_to_binary(value: int, length: int = 4) -> str:
    """Convert a decimal value to binary with fixed length."""
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
    """Prompt for input, validate, parse, and return a list of values of the expected type."""
    while True:
        logging.debug(f"Validating and parsing input for prompt: {prompt}")
        response = await input_optional(prompt)
        if optional and response is None:
            logging.debug("Optional input not provided")
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
    """Construct the bit matrix representation."""
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


async def main():
    """Main function to drive the program."""
    try:
        logging.info("Starting main function")
        result = await prompt_for_values()
        # Ensure you await the coroutine and then unpack or use its result
        version, channels, counts, signs, figures, values, uncertainties = result
        bit_matrix = await construct_bit_matrix(version, channels, counts, signs, figures, values, uncertainties)
        print(f"Constructed Bit Matrix Representation: {bit_matrix}")
    except Exception as e:
        logging.exception(f"Error in main function: {e}")
        raise e


# Web handler for bit matrix construction
async def construct_matrix_handler(request):
    try:
        data = await request.json()
        version = data.get("version", 1)
        channels = data.get("channels", [1, 0, 0, 0, 0, 0, 0, 0, 0])
        counts = data.get("counts", [0])
        signs = data.get("signs", [1])
        figures = data.get("figures", 6)
        values = data.get("values", [0])
        uncertainties = data.get("uncertainties", [0])

        bit_matrix = await construct_bit_matrix(
            version, channels, counts, signs, figures, values, uncertainties
        )
        return web.json_response({"bit_matrix": bit_matrix})
    except Exception as e:
        logging.exception(f"Error in construct matrix handler: {e}")
        return web.json_response({"error": str(e)}, status=400)


app.add_routes([web.post("/construct_matrix", construct_matrix_handler)])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    web.run_app(app)
