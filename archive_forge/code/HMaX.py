class ResourceProfiler:
    """

    Profiles resource usage (CPU, memory, etc.) and logs the metrics at specified intervals.

    """

    def __init__(self, interval: int = 60, output: str = "resource_usage.log"):
        """

        Initializes the resource profiler.

        Args:

            interval (int): The interval (in seconds) at which to log resource usage.

            output (str): The file path for resource profiling logs.

        """

        self.interval = interval

        self.output = output

        self.running = False

        self.lock = asyncio.Lock()  # Ensures thread safety in async context

    async def start(self):
        """

        Starts the resource profiling in a background coroutine.

        """

        async with self.lock:

            if not self.running:

                self.running = True

                tracemalloc.start()

                asyncio.create_task(self._profile())

    async def stop(self):
        """

        Stops the resource profiling.

        """

        async with self.lock:

            if self.running:

                self.running = False

                tracemalloc.stop()

    async def _profile(self):
        """

        Periodically logs resource usage until stopped.

        """

        while self.running:

            await self._log_resource_usage()

            await asyncio.sleep(self.interval)

    async def _log_resource_usage(self):
        """

        Logs the current resource usage asynchronously.

        """

        process = psutil.Process()

        cpu_usage = psutil.cpu_percent()

        memory_usage = process.memory_percent()

        current, peak = tracemalloc.get_traced_memory()

        async with self.lock:

            async with aiofiles.open(self.output, "a") as af:

                await af.write(
                    f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes\n"
                )

        logging.info(
            f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, Traced Memory: Current {current} bytes, Peak {peak} bytes"
        )
