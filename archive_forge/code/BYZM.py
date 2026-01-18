import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple
import asyncio

import async_file_handler as afh


class AsyncCache:
    def __init__(
        self,
        cache_lifetime: int = 60,
        cache_maxsize: int = 128,
        enable_caching: bool = True,
        file_cache_path: str = "file_cache.pkl",
        call_threshold: int = 10,
        cache_key_strategy: Optional[Callable] = None,
    ):
        """
        Initializes the AsyncCache instance with the specified parameters.

        Args:
            cache_lifetime (int): The lifetime of cache entries in seconds. Defaults to 60.
            cache_maxsize (int): The maximum size of the in-memory cache. Defaults to 128.
            enable_caching (bool): A flag to enable or disable caching. Defaults to True.
            file_cache_path (str): The path to the file cache. Defaults to "file_cache.pkl".
            call_threshold (int): The threshold for saving cache entries to the file cache. Defaults to 10.
            cache_key_strategy (Optional[Callable]): A custom cache key generation strategy. Defaults to None.
        """
        self.cache_lifetime = cache_lifetime
        self.cache_maxsize = cache_maxsize
        self.enable_caching = enable_caching
        self.file_cache_path = file_cache_path
        self.call_threshold = call_threshold
        self.cache_key_strategy = cache_key_strategy or self._default_cache_key_strategy
        self.cache = {}
        self.file_cache = {}
        self.cache_lock = asyncio.Lock()
        self.file_io_lock = asyncio.Lock()
        self.call_counter = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(afh.AsyncFileHandler("cache.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._initialize_cache())
        self.loop.create_task(self.maintain_cache())

    def _default_cache_key_strategy(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, ...]:
        """
        Default cache key strategy that generates a cache key based on the function arguments.
        """
        return self.generate_cache_key(func, *args, **kwargs)

    async def _initialize_cache(self):
        """
        Initializes caching mechanisms based on configuration asynchronously.
        """
        if self.enable_caching:
            # Initialize in-memory cache
            self.cache = {}
            # Check and initialize file-based cache if specified
            if os.path.exists(self.file_cache_path):
                async with self.file_io_lock:
                    async with afh.open(self.file_cache_path, "rb") as f:
                        self.cache = await f.read()
                        self.cache = pickle.loads(self.cache)
            self.logger.debug("Caching mechanisms initialized asynchronously.")

    async def background_cache_persistence(self, key, value):
        """
        Initiates an asynchronous task to persist cache updates to the file cache asynchronously.
        This method is used to ensure non-blocking operations and thread safety in an asyncio context.
        """
        try:
            await asyncio.create_task(self._save_to_file_cache_async(key, value))
        except Exception as e:
            logging.error(
                f"Error initiating background cache persistence for key {key}: {e}"
            )

    def generate_cache_key(
        self, func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """
        Generates a unique cache key based on the function name, arguments, and keyword arguments.
        This method serializes the arguments and keyword arguments to ensure uniqueness.
        """
        sorted_kwargs = tuple(sorted(kwargs.items()))
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(sorted_kwargs)
        cache_key = (func.__name__, serialized_args, serialized_kwargs)
        logging.debug(f"Generated cache key: {cache_key} for function {func.__name__}")
        return cache_key

    async def cache_logic(
        self, key: Tuple[Any, ...], func: Callable, *args, **kwargs
    ) -> Any:
        """
        Implements the caching logic, checking for cache hits in both in-memory and file caches.
        If a cache miss occurs, the function is executed and the result is cached.
        This method handles both synchronous and asynchronous functions dynamically.
        """
        async with self.cache_lock:
            try:
                # Check in-memory cache first
                if key in self.cache:
                    # Move the key to the end to mark it as recently used
                    self.cache.move_to_end(key)
                    self.call_counter[key] += 1
                    logging.debug(f"Cache hit for {key}.")
                    return self.cache[key][
                        0
                    ]  # Return the result part of the cache entry
                # Check file cache next
                elif key in self.file_cache:
                    result = self.file_cache[key]
                    logging.debug(f"File cache hit for {key}.")
                else:
                    # Handle cache miss
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)
                    logging.debug(
                        f"Cache miss for {key}. Function executed and result cached."
                    )
                    # Update in-memory cache
                    self.cache[key] = (
                        result,
                        datetime.utcnow() + timedelta(seconds=self.cache_lifetime),
                    )
                    self.call_counter[key] = 1
                    # Evict least recently used item if cache exceeds max size
                    if len(self.cache) > self.cache_maxsize:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        del self.call_counter[oldest_key]
                        logging.debug(
                            f"Evicted least recently used cache item: {oldest_key}"
                        )
                    # Save cache updates to file asynchronously
                    await self.background_cache_persistence(key, self.cache[key])
                    logging.debug(f"Result cached for key: {key}")
                    return result
            except Exception as e:
                logging.error(f"Error in cache logic for key {key}: {e}")
                raise

    async def _load_file_cache(self) -> None:
        """
        Asynchronously loads the file cache from the specified file cache path, ensuring non-blocking I/O operations.
        This method checks the existence of the file cache path and loads the cache using asynchronous file operations,
        thereby preventing the blocking of the event loop and maintaining the responsiveness of the application.
        The method employs a try-except block to gracefully handle any exceptions that may arise during the file
        operations, logging the error details for troubleshooting while ensuring the robustness of the cache loading process.
        Raises:
            Exception: Logs an error message indicating the failure to load the file cache due to an encountered exception.
        """
        try:
            if os.path.exists(self.file_cache_path):
                async with self.file_io_lock:
                    async with afh.open(self.file_cache_path, mode="rb") as file:
                        file_content = await file.read()
                        # Utilizing asyncio.to_thread to offload the blocking operation, pickle.loads, to a separate thread.
                        self.file_cache = await asyncio.to_thread(
                            pickle.loads, file_content
                        )
                        logging.debug(
                            f"File cache successfully loaded from {self.file_cache_path}."
                        )
            else:
                # Initializing an empty cache if the file cache does not exist.
                self.file_cache = {}
                logging.debug("File cache not found. Initialized an empty cache.")
        except Exception as error:
            # Logging the exception details in case of failure to load the file cache.
            logging.error(f"Failed to load file cache due to error: {error}")
            raise

    async def _save_file_cache(self) -> None:
        """
        Asynchronously saves the current state of the file cache to disk, leveraging asynchronous file operations
        to prevent event loop blocking. This method encapsulates the file writing operation within a try-except block
        to gracefully handle potential exceptions, ensuring the application's robustness and reliability.
        The method utilizes the asyncio.to_thread function to offload the blocking operation, pickle.dump, to a separate
        thread, thereby maintaining the responsiveness of the application by not blocking the event loop.

        Raises:
            Exception: Logs an error message indicating the failure to save the file cache due to an encountered exception.
        """
        try:
            # Serialize the file cache first to avoid blocking in async context
            serialized_cache = await asyncio.to_thread(
                pickle.dumps, self.file_cache, pickle.HIGHEST_PROTOCOL
            )
            async with self.file_io_lock:
                async with afh.open(self.file_cache_path, mode="wb") as file:
                    # Asynchronously write the serialized cache to the file
                    await file.write(serialized_cache)
                    logging.debug("File cache has been successfully saved to disk.")
        except Exception as error:
            # Logging the exception details in case of failure to save the file cache.
            logging.error(f"Failed to save file cache due to error: {error}")
            raise

    async def clean_expired_cache_entries(self):
        """
        Cleans expired entries from both in-memory and file caches asynchronously.
        This method iterates over the cache entries and removes those that have expired,
        ensuring thread safety by utilizing an asyncio.Lock.
        """
        async with self.cache_lock:
            try:
                current_time = datetime.utcnow()
                # Cleaning in-memory cache
                expired_keys = [
                    key
                    for key, (_, expiration_time) in self.cache.items()
                    if expiration_time < current_time
                ]
                for key in expired_keys:
                    del self.cache[key]
                    logging.debug(f"Removed expired cache entry: {key}")

                # Cleaning file cache
                expired_keys_file_cache = [
                    key
                    for key, (_, expiration_time) in self.file_cache.items()
                    if expiration_time < current_time
                ]
                for key in expired_keys_file_cache:
                    del self.file_cache[key]
                if expired_keys_file_cache:
                    await self._save_file_cache()
                logging.debug(
                    f"Expired entries removed from file cache: {expired_keys_file_cache}"
                )
            except Exception as e:
                logging.error(f"Error cleaning expired cache entries: {e}")
                raise

    async def _evict_lru_from_memory(self):
        """
        Asynchronously evicts the least recently used (LRU) item from the in-memory cache.
        This method ensures thread safety by utilizing an asyncio.Lock, preventing race conditions
        during the eviction process. It logs detailed information about the eviction process,
        including the key of the evicted item, to aid in debugging and monitoring of cache behavior.

        Raises:
            Exception: Logs an error message indicating the failure to evict the LRU item due to an encountered exception.
        """
        async with self.cache_lock:
            try:
                if len(self.cache) > self.cache_maxsize:
                    # Identify the least recently used item
                    lru_key = next(iter(self.cache))
                    # Evict the identified LRU item
                    del self.cache[lru_key]
                    del self.call_counter[lru_key]
                    logging.debug(f"Evicted least recently used cache item: {lru_key}")

                    # Asynchronously save the updated cache state to the file cache
                    await self._save_file_cache()

                    logging.info(
                        f"Successfully evicted LRU item from memory: {lru_key} and updated file cache."
                    )
                else:
                    logging.debug("No eviction needed. Cache size within limits.")
            except Exception as e:
                logging.error(f"Error evicting LRU item from memory: {e}")
                raise

    async def maintain_cache(self):
        """
        Periodically checks and maintains the cache by cleaning expired entries and evicting LRU items.
        This method runs in a separate asynchronous task to ensure efficient cache management.
        """
        try:
            # This enhanced block continuously monitors and maintains the cache, incorporating advanced logic to
            # determine the program's execution context and initiate a graceful shutdown if the cache remains unchanged
            # for an extended period, indicating potential program inactivity or improper closure.

            # Initialize variables to track cache changes and manage shutdown logic
            unchanged_cycles = (
                0  # Counter for the number of cycles with no cache changes
            )
            is_main_program = (
                __name__ == "__main__"
            )  # Check if running as the main program
            sleep_duration = (
                60 if is_main_program else 10
            )  # Sleep duration based on execution context

            # Continuously perform cache maintenance tasks with advanced monitoring for inactivity
            while True:
                await asyncio.sleep(
                    sleep_duration
                )  # Pause execution for the determined duration

                # Capture the state of caches before maintenance to detect changes
                initial_memory_cache_state = str(self.cache)
                initial_file_cache_state = str(self.file_cache)

                # Execute cache maintenance operations
                await self.clean_expired_cache_entries()  # Clean expired entries from caches
                await self._evict_lru_from_memory()  # Evict least recently used items from memory cache

                # Determine if the cache states have changed following maintenance operations
                memory_cache_changed = initial_memory_cache_state != str(self.cache)
                file_cache_changed = initial_file_cache_state != str(self.file_cache)

                # Log the completion of cache maintenance based on the program's execution context
                context_message = (
                    "the main program"
                    if is_main_program
                    else "the imported module context"
                )
                logging.debug(f"Cache maintenance completed in {context_message}.")

                # Update the unchanged cycles counter based on cache state changes
                if memory_cache_changed or file_cache_changed:
                    unchanged_cycles = 0  # Reset counter if changes were detected
                else:
                    unchanged_cycles += (
                        1  # Increment counter if no changes were detected
                    )

                # Initiate a graceful shutdown if caches have remained unchanged for 10 cycles
                if unchanged_cycles >= 10:
                    logging.info(
                        "No cache changes detected for 10 cycles. Initiating graceful shutdown."
                    )
                    # Placeholder for graceful shutdown logic
                    # This should include tasks such as closing database connections, stopping async tasks,
                    # and any other cleanup required before safely terminating the program.
                    await self.initiate_graceful_shutdown()
                    break  # Exit the loop to stop further cache maintenance

        except Exception as e:
            # Log any exceptions encountered during the cache maintenance and monitoring process
            logging.error(f"Error in cache maintenance task: {e}")

    async def attempt_cache_retrieval(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """
        Attempts to retrieve the cached result for the given key from both in-memory and file caches.
        This method checks the caches and returns the cached result if found.
        """
        try:
            async with self.cache_lock:
                if key in self.cache:
                    logging.debug(f"In-memory cache hit for {key}")
                    return self.cache[key]
                elif key in self.file_cache:
                    logging.debug(f"File cache hit for {key}")
                    result = self.file_cache[key]
                    if len(self.cache) >= self.cache_maxsize:
                        await self._evict_lru_from_memory()
                    self.cache[key] = result
                    return result
            return None
        except Exception as e:
            logging.error(f"Error attempting cache retrieval for key {key}: {e}")
            return None

    async def update_cache(
        self, key: Tuple[Any, ...], result: Any, is_async: bool = True
    ):
        """
        Updates the cache with the given key and result, managing cache sizes, evictions, and time-based expiration.
        This method handles both in-memory and file cache updates, including time-based eviction.
        """
        try:
            current_time = datetime.utcnow()
            expiration_time = current_time + timedelta(seconds=self.cache_lifetime)
            async with self.cache_lock:
                if len(self.cache) >= self.cache_maxsize:
                    await self._evict_lru_from_memory()
                self.cache[key] = (result, expiration_time)
                if self.call_counter.get(key, 0) >= self.call_threshold:
                    self.call_counter[key] = 0
                    if is_async:
                        await self._save_to_file_cache(key, (result, expiration_time))
                else:
                    await self.background_cache_persistence(
                        key, (result, expiration_time)
                    )
                    logging.debug(
                        f"Cache updated for key: {key}. Cache size: {len(self.cache)}"
                    )
        except Exception as e:
            logging.error(f"Error updating cache for key {key}: {e}")

    async def _save_to_file_cache(
        self, key: Tuple[Any, ...], value: Tuple[Any, datetime]
    ):
        """
        Asynchronously saves a key-value pair to the file cache.
        This method uses asynchronous file operations to ensure the event loop is not blocked.
        """
        try:
            async with self.cache_lock:
                self.file_cache[key] = value
                async with self.file_io_lock:
                    async with aiofiles.open(self.file_cache_path, "wb") as f:
                        await f.write(
                            pickle.dumps(self.file_cache, pickle.HIGHEST_PROTOCOL)
                        )
            logging.debug(f"File cache asynchronously updated for key: {key}")
        except Exception as e:
            logging.error(
                f"Error saving to file cache asynchronously for key {key}: {e}"
            )
