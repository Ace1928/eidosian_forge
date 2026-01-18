import asyncio
import logging
import tracemalloc
import memory_profiler
import cProfile
import pstats
import psutil
import aiofiles  # Correcting the missing import based on lint_context_0
from typing_extensions import NoReturn
from typing import Optional, Literal, Union
from async_file_handler import (
    AsyncFileHandler,
)  # Ensuring explicit import for clarity and maintainability

# Extending the logging capabilities by integrating a custom async file handler for non-blocking logging operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(AsyncFileHandler("resource_profiler.log"))
logger.addHandler(
    logging.StreamHandler()
)  # Ensuring console output for immediate visibility
logger.propagate = False  # Preventing duplicate logging in case of multiple handlers


class ResourceProfiler:
    """
    A comprehensive class dedicated to profiling and logging various system resources such as CPU, memory, and traced memory.
    This class utilizes asynchronous programming paradigms to ensure non-blocking operations, enhancing the performance
    and scalability of applications that require meticulous resource monitoring.

    Attributes:
        interval (int): The interval, in seconds, at which resource usage metrics are logged.
        output (str): The file path where resource usage metrics are persisted.
        running (bool): A flag indicating whether the resource profiling is currently active.
        lock (asyncio.Lock): An asyncio lock to ensure thread-safe operations in an asynchronous context.
    """

    def __init__(self, interval: int = 60, output: str = "resource_usage.log") -> None:
        """
        Initializes the ResourceProfiler with specified logging interval and output file path.

        Args:
            interval (int): The interval, in seconds, at which to log resource usage metrics. Defaults to 60 seconds.
            output (str): The file path for persisting resource profiling logs. Defaults to 'resource_usage.log'.
        """
        self.interval: int = interval
        self.output: str = output
        self.running: bool = False
        self.lock: asyncio.Lock = (
            asyncio.Lock()
        )  # Ensuring thread safety in async context

    async def start(self) -> NoReturn:
        """
        Initiates the resource profiling process in a non-blocking manner.
        This method starts the memory tracing and schedules the periodic resource usage logging.
        """
        async with self.lock:
            if not self.running:
                self.running = True
                tracemalloc.start()
                logger.debug("Resource profiling started.")
                asyncio.create_task(self._profile())

    async def stop(self) -> NoReturn:
        """
        Terminates the resource profiling process.
        This method stops the memory tracing and halts the periodic resource usage logging.
        """
        async with self.lock:
            if self.running:
                self.running = False
                tracemalloc.stop()
                logger.debug("Resource profiling stopped.")

    async def _profile(self) -> NoReturn:
        """
        Periodically logs resource usage metrics until the profiling is stopped.
        This method runs in an infinite loop, broken only when the `running` flag is set to False.
        """
        while self.running:
            await self._log_resource_usage()
            await asyncio.sleep(self.interval)

    async def _log_resource_usage(self) -> NoReturn:
        """
        Asynchronously logs the current resource usage metrics, including CPU, memory, and traced memory.
        This method gathers resource usage metrics and writes them to both the specified output file and the log.
        """
        process = psutil.Process()
        cpu_usage = psutil.cpu_percent()
        memory_usage = process.memory_percent()
        current, peak = tracemalloc.get_traced_memory()

        log_message = (
            f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, "
            f"Traced Memory: Current {current} bytes, Peak {peak} bytes"
        )

        async with self.lock:
            async with aiofiles.open(self.output, "a") as file:
                await file.write(f"{log_message}\n")

        logger.info(log_message)
