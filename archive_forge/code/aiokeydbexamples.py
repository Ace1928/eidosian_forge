import time
import asyncio
import uuid
from aiokeydb import KeyDBClient

# The session can be explicitly initialized, or
# will be lazily initialized on first use
# through environment variables with all
# params being prefixed with `KEYDB_`

keydb_uri = "keydb://localhost:6379/0"

# Initialize the Unified Client
KeyDBClient.init_session(
    uri=keydb_uri,
)

# Cache the results of these functions
# cachify works for both sync and async functions
# and has many params to customize the caching behavior
# and supports both `redis` and `keydb` backends
# as well as `api` frameworks such as `fastapi` and `starlette`


@KeyDBClient.cachify()
async def async_fibonacci(number: int):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    return await async_fibonacci(number - 1) + await async_fibonacci(number - 2)


@KeyDBClient.cachify()
def fibonacci(number: int):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    return fibonacci(number - 1) + fibonacci(number - 2)


async def test_fib(n: int = 100, runs: int = 10):
    # Test that both results are the same.
    sync_t, async_t = 0.0, 0.0

    for i in range(runs):
        t = time.time()
        print(f"[Async - {i}/{runs}] Fib Result: {await async_fibonacci(n)}")
        tt = time.time() - t
        print(f"[Async - {i}/{runs}] Fib Time: {tt:.2f}s")
        async_t += tt

        t = time.time()
        print(f"[Sync  - {i}/{runs}] Fib Result: {fibonacci(n)}")
        tt = time.time() - t
        print(f"[Sync  - {i}/{runs}] Fib Time: {tt:.2f}s")
        sync_t += tt

    print(
        f"[Async] Cache Average Time: {async_t / runs:.2f}s | Total Time: {async_t:.2f}s"
    )
    print(
        f"[Sync ] Cache Average Time: {sync_t / runs:.2f}s | Total Time: {sync_t:.2f}s"
    )


async def test_setget(runs: int = 10):
    # By default, the client utilizes `pickle` to serialize
    # and deserialize objects. This can be changed by setting
    # the `serializer`

    sync_t, async_t = 0.0, 0.0
    for i in range(runs):
        value = str(uuid.uuid4())
        key = f"async-test-{i}"
        t = time.time()
        await KeyDBClient.async_set(key, value)
        assert await KeyDBClient.async_get(key) == value
        tt = time.time() - t
        print(f"[Async - {i}/{runs}] Get/Set: {key} -> {value} = {tt:.2f}s")
        async_t += tt

        value = str(uuid.uuid4())
        key = f"sync-test-{i}"
        t = time.time()
        KeyDBClient.set(key, value)
        assert KeyDBClient.get(key) == value
        tt = time.time() - t
        print(f"[Sync  - {i}/{runs}] Get/Set: {key}  -> {value} = {tt:.2f}s")
        sync_t += tt

    print(
        f"[Async] GetSet Average Time: {async_t / runs:.2f}s | Total Time: {async_t:.2f}s"
    )
    print(
        f"[Sync ] GetSet Average Time: {sync_t / runs:.2f}s | Total Time: {sync_t:.2f}s"
    )


async def run_tests(fib_n: int = 100, fib_runs: int = 10, setget_runs: int = 10):

    # You can explicitly wait for the client to be ready
    # Sync version
    # KeyDBClient.wait_for_ready()
    await KeyDBClient.async_wait_for_ready()

    # Run the tests
    await test_fib(n=fib_n, runs=fib_runs)
    await test_setget(runs=setget_runs)

    # Utilize the current session
    await KeyDBClient.async_set("async_test_0", "test")
    assert await KeyDBClient.async_get("async_test_0") == "test"

    KeyDBClient.set("sync_test_0", "test")
    assert KeyDBClient.get("sync_test_0") == "test"

    # you can access the `KeyDBSession` object directly
    # which mirrors the APIs in `KeyDBClient`

    await KeyDBClient.session.async_set("async_test_1", "test")
    assert await KeyDBClient.session.async_get("async_test_1") == "test"

    KeyDBClient.session.set("sync_test_1", "test")
    assert KeyDBClient.session.get("sync_test_1") == "test"

    # The underlying client can be accessed directly
    # if the desired api methods aren't mirrored

    # KeyDBClient.keydb
    # KeyDBClient.async_keydb
    # Since encoding / decoding is not handled by the client
    # you must encode / decode the data yourself
    await KeyDBClient.async_keydb.set("async_test_2", b"test")
    assert await KeyDBClient.async_keydb.get("async_test_2") == b"test"

    KeyDBClient.keydb.set("sync_test_2", b"test")
    assert KeyDBClient.keydb.get("sync_test_2") == b"test"

    # You can also explicitly close the client
    # However, this closes the connectionpool and will terminate
    # all connections. This is not recommended unless you are
    # explicitly closing the client.

    # Sync version
    # KeyDBClient.close()
    await KeyDBClient.aclose()


asyncio.run(run_tests())


from aiokeydb import KeyDB, AsyncKeyDB, from_url

sync_client = KeyDB()
async_client = AsyncKeyDB()

# Alternative methods
sync_client = from_url("keydb://localhost:6379/0")
async_client = from_url("keydb://localhost:6379/0", asyncio=True)


from aiokeydb import KeyDBClient
from lazyops.utils import logger

keydb_uris = {
    "default": "keydb://127.0.0.1:6379/0",
    "cache": "keydb://localhost:6379/1",
    "public": "redis://public.redis.db:6379/0",
}

sessions = {}

# these will now be initialized
for name, uri in keydb_uris.items():
    sessions[name] = KeyDBClient.init_session(
        name=name,
        uri=uri,
    )
    logger.info(f"Session {name}: uri: {sessions[name].uri}")

# however if you initialize another session
# it will use the global environment vars

sessions["test"] = KeyDBClient.init_session(
    name="test",
)
logger.info(f'Session test: uri: {sessions["test"].uri}')

from aiokeydb import KeyDBClient
from lazyops.utils import logger

default_uri = "keydb://public.host.com:6379/0"
keydb_dbs = {
    "cache": {
        "db_id": 1,
    },
    "db": {
        "uri": "keydb://127.0.0.1:6379/0",
    },
}

KeyDBClient.configure(
    url=default_uri,
    debug_enabled=True,
    queue_db=1,
)

# now any sessions that are initialized will use the global settings

sessions = {}
# these will now be initialized

# Initialize the first default session
# which should utilize the `default_uri`
KeyDBClient.init_session()

for name, config in keydb_dbs.items():
    sessions[name] = KeyDBClient.init_session(name=name, **config)
    logger.info(f"Session {name}: uri: {sessions[name].uri}")


import asyncio
from aiokeydb import KeyDBClient
from aiokeydb.queues import TaskQueue, Worker
from lazyops.utils import logger

# Configure the KeyDB Client - the default keydb client will use
# db = 0, and queue uses 2 so that it doesn't conflict with other
# by configuring it here, you can explicitly set the db to use
keydb_uri = "keydb://127.0.0.1:6379/0"
# Configure the Queue to use db = 1 instead of 2
KeyDBClient.configure(
    url=keydb_uri,
    debug_enabled=True,
    queue_db=1,
)


@Worker.add_cronjob("*/1 * * * *")
async def test_cron_task(*args, **kwargs):
    logger.info("Cron task ran")
    await asyncio.sleep(5)


@Worker.add_function()
async def test_task(*args, **kwargs):
    logger.info("Task ran")
    await asyncio.sleep(5)


async def run_tests():
    queue = TaskQueue("test_queue")
    worker = Worker(queue)
    await worker.start()


asyncio.run(run_tests())


import asyncio
import time
import uuid
from typing import Any, Dict
from aiokeydb import KeyDBClient, AsyncKeyDB
import logging

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# KeyDB Configuration
KEYDB_URI = "keydb://localhost:6379/0"


# Initialize the Unified Client with KeyDB
async def init_keydb_client():
    """
    Initializes the KeyDB client session with the specified URI.
    This function ensures that the KeyDB client is ready for asynchronous operations.
    """
    await KeyDBClient.init_session(uri=KEYDB_URI)
    logging.info("KeyDB client session initialized.")


# Example usage of KeyDB for caching with the cachify decorator
@KeyDBClient.cachify()
async def async_fibonacci(number: int) -> int:
    """
    Asynchronously calculates the Fibonacci number using recursion with caching.
    The results are cached to improve performance on subsequent calls with the same input.
    Args:
        number (int): The position in the Fibonacci sequence.
    Returns:
        int: The Fibonacci number at the specified position.
    """
    if number in (0, 1):
        return number
    return await async_fibonacci(number - 1) + await async_fibonacci(number - 2)


@KeyDBClient.cachify()
def fibonacci(number: int) -> int:
    """
    Synchronously calculates the Fibonacci number using recursion with caching.
    The results are cached to improve performance on subsequent calls with the same input.
    Args:
        number (int): The position in the Fibonacci sequence.
    Returns:
        int: The Fibonacci number at the specified position.
    """
    if number in (0, 1):
        return number
    return fibonacci(number - 1) + fibonacci(number - 2)


async def test_fib(n: int = 100, runs: int = 10):
    """
    Tests and compares the performance of synchronous and asynchronous Fibonacci functions.
    Args:
        n (int): The position in the Fibonacci sequence to calculate.
        runs (int): The number of times to run the test for averaging performance.
    """
    sync_t, async_t = 0.0, 0.0
    for i in range(runs):
        t = time.time()
        fib_result = await async_fibonacci(n)
        tt = time.time() - t
        logging.info(
            f"[Async - {i}/{runs}] Fib Result: {fib_result}, Fib Time: {tt:.2f}s"
        )
        async_t += tt
        t = time.time()
        fib_result = fibonacci(n)
        tt = time.time() - t
        logging.info(
            f"[Sync  - {i}/{runs}] Fib Result: {fib_result}, Fib Time: {tt:.2f}s"
        )
        sync_t += tt
    logging.info(
        f"[Async] Cache Average Time: {async_t / runs:.2f}s | Total Time: {async_t:.2f}s"
    )
    logging.info(
        f"[Sync ] Cache Average Time: {sync_t / runs:.2f}s | Total Time: {sync_t:.2f}s"
    )


async def test_setget(runs: int = 10):
    """
    Tests the performance of setting and getting values asynchronously and synchronously in KeyDB.

    Args:
        runs (int): The number of times to run the set/get operations for averaging performance.
    """
    sync_t, async_t = 0.0, 0.0
    for i in range(runs):
        value = str(uuid.uuid4())
        key = f"async-test-{i}"
        t = time.time()
        await KeyDBClient.async_set(key, value)
        assert await KeyDBClient.async_get(key) == value, "Async set/get mismatch"
        tt = time.time() - t
        logging.info(f"[Async - {i}/{runs}] Get/Set: {key} -> {value} = {tt:.2f}s")
        async_t += tt

        value = str(uuid.uuid4())
        key = f"sync-test-{i}"
        t = time.time()
        KeyDBClient.set(key, value)
        assert KeyDBClient.get(key) == value, "Sync set/get mismatch"
        tt = time.time() - t
        logging.info(f"[Sync - {i}/{runs}] Get/Set: {key} -> {value} = {tt:.2f}s")
        sync_t += tt
        logging.info(
            f"[Async] GetSet Average Time: {async_t / runs:.2f}s | Total Time: {async_t:.2f}s"
        )
        logging.info(
            f"[Sync ] GetSet Average Time: {sync_t / runs:.2f}s | Total Time: {sync_t:.2f}s"
        )


async def run_tests(fib_n: int = 100, fib_runs: int = 10, setget_runs: int = 10):
    """
    Runs a series of tests to evaluate the performance and functionality of the KeyDB client.
    This includes Fibonacci calculations and basic set/get operations.
    Args:
    fib_n (int): The position in the Fibonacci sequence to calculate for the test.
    fib_runs (int): The number of runs for the Fibonacci test.
    setget_runs (int): The number of runs for the set/get test.
    """
    # Ensure the KeyDB client is ready
    await KeyDBClient.async_wait_for_ready()
    # Run Fibonacci tests
    await test_fib(n=fib_n, runs=fib_runs)
    # Run set/get tests
    await test_setget(runs=setget_runs)
    # Demonstrate direct session access for set/get operations
    await KeyDBClient.async_set("async_test_0", "test")
    assert (
        await KeyDBClient.async_get("async_test_0") == "test"
    ), "Async direct session set/get mismatch"
    KeyDBClient.set("sync_test_0", "test")
    assert (
        KeyDBClient.get("sync_test_0") == "test"
    ), "Sync direct session set/get mismatch"
    await KeyDBClient.session.async_set("async_test_1", "test")
    assert (
        await KeyDBClient.session.async_get("async_test_1") == "test"
    ), "Async session set/get mismatch"


if name == "main":
    asyncio.run(run_tests())




# Ensure that the KeyDB client and sessions are properly initialized and managed asynchronously
# This includes handling connection errors, session conflicts, and ensuring that each session
# is configured with the correct URI from the extensive list of available URIs for different cache purposes.
T = TypeVar("T")
KEYDB_CACHE_URIS = {
    "development_blob": "keydb://localhost:6379/2",
    "public_blob": "keydb://localhost:6379/4",
    "stock_cache": "keydb://localhost:6379/8",
    "user_cache": "keydb://localhost:6379/10",
    "log_cache": "keydb://localhost:6379/12",
    "media_cache": "keydb://localhost:6379/14",
    "queue_cache": "keydb://localhost:6379/16",
    "file_cache": "keydb://localhost:6379/18",
    "logging_cache": "keydb://localhost:6379/20",
    "process_cache": "keydb://localhost:6379/22",
    "function_cache": "keydb://localhost:6379/24",
    "method_cache": "keydb://localhost:6379/26",
    "class_cache": "keydb://localhost:6379/28",
    "module_cache": "keydb://localhost:6379/30",
    "package_cache": "keydb://localhost:6379/32",
    "argument_cache": "keydb://localhost:6379/34",
    "result_cache": "keydb://localhost:6379/36",
    "exception_cache": "keydb://localhost:6379/38",
    "input_cache": "keydb://localhost:6379/40",
    "intermediate_result_cache": "keydb://localhost:6379/42",
    "output_cache": "keydb://localhost:6379/44",
    "context_cache": "keydb://localhost:6379/46",
    "profile_cache": "keydb://localhost:6379/48",
    "monitor_cache": "keydb://localhost:6379/50",
    "metrics_cache": "keydb://localhost:6379/52",
    "log_cache": "keydb://localhost:6379/54",
    "error_log_cache": "keydb://localhost:6379/56",
    "warning_log_cache": "keydb://localhost:6379/58",
    "info_log_cache": "keydb://localhost:6379/60",
    "debug_log_cache": "keydb://localhost:6379/62",
    "trace_log_cache": "keydb://localhost:6379/64",
    "standards_cache": "keydb://localhost:6379/66",
    "guidelines_cache": "keydb://localhost:6379/68",
    "requirements_cache": "keydb://localhost:6379/70",
    "specifications_cache": "keydb://localhost:6379/72",
    "documentation_cache": "keydb://localhost:6379/74",
    "logic_cache": "keydb://localhost:6379/76",
    "flow_cache": "keydb://localhost:6379/78",
    "control_cache": "keydb://localhost:6379/80",
    "data_cache": "keydb://localhost:6379/82",
    "state_cache": "keydb://localhost:6379/84",
    "operations_cache": "keydb://localhost:6379/86",
    "bytestream_cache": "keydb://localhost:6379/88",
    "bit_matrix_cache": "keydb://localhost:6379/90",
    "vector_cache": "keydb://localhost:6379/92",
KEYDB_BLOB_URIS = {
    "development_blob": "keydb://localhost:6479/2",
    "public_blob": "keydb://localhost:6479/4",
    "stock_blob": "keydb://localhost:6479/8",
    "user_blob": "keydb://localhost:6479/10",
    "log_blob": "keydb://localhost:6479/12",
    "media_blob": "keydb://localhost:6479/14",
    "queue_blob": "keydb://localhost:6479/16",
    "file_blob": "keydb://localhost:6479/18",
    "logging_blob": "keydb://localhost:6479/20",
    "process_blob": "keydb://localhost:6479/22",
    "function_blob": "keydb://localhost:6479/24",
    "method_blob": "keydb://localhost:6479/26",
    "class_blob": "keydb://localhost:6479/28",
    "module_blob": "keydb://localhost:6479/30",
    "package_blob": "keydb://localhost:6479/32",
    "argument_blob": "keydb://localhost:6479/34",
    "result_blob": "keydb://localhost:6479/36",
    "exception_blob": "keydb://localhost:6479/38",
    "input_blob": "keydb://localhost:6479/40",
    "intermediate_result_blob": "keydb://localhost:6479/42",
    "output_blob": "keydb://localhost:6479/44",
    "context_blob": "keydb://localhost:6479/46",
    "profile_blob": "keydb://localhost:6479/48",
    "monitor_blob": "keydb://localhost:6479/50",
    "metrics_blob": "keydb://localhost:6479/52",
    "log_blob": "keydb://localhost:6479/54",
    "error_log_blob": "keydb://localhost:6479/56",
    "warning_log_blob": "keydb://localhost:6479/58",
    "info_log_blob": "keydb://localhost:6479/60",
    "debug_log_blob": "keydb://localhost:6479/62",
    "trace_log_blob": "keydb://localhost:6479/64",
    "standards_blob": "keydb://localhost:6479/66",
    "guidelines_blob": "keydb://localhost:6479/68",
    "requirements_blob": "keydb://localhost:6479/70",
    "specifications_blob": "keydb://localhost:6479/72",
    "documentation_blob": "keydb://localhost:6479/74",
    "logic_blob": "keydb://localhost:6479/76",
    "flow_blob": "keydb://localhost:6479/78",
    "control_blob": "keydb://localhost:6479/80",
    "data_blob": "keydb://localhost:6479/82",
    "state_blob": "keydb://localhost:6479/84",
    "operations_blob": "keydb://localhost:6479/86",
    "bytestream_blob": "keydb://localhost:6479/88",
    "bit_matrix_blob": "keydb://localhost:6479/90",
    "vector_blob": "keydb://localhost:6479/92",
KeyDBClient.configure(
    url=KEYDB_CACHE_URIS["default_blob"],
    debug_enabled=True,
    logger=logger,
    queue_db=1,
    queue_blob=1,
    queue_cache=1,
    queue_media=1,
    queue_log=1,
    queue_error_log=1,
    queue_warning_log=1,
    queue_info_log=1,
    queue_debug_log=1,
    queue_trace_log=1,
    queue_metrics=1,
    queue_monitor=1,
    queue_profile=1,
    queue_input=1,
    queue_output=1,
    queue_session=1,
)
sessions = {}
sessions["test"] = KeyDBClient.init_session(
    name="test",
)
logger.info(f'Session test: uri: {sessions["test"].uri}')
# Initialize the Unified Client with KeyDB
async def init_keydb_client():
    Initializes the KeyDB client session with the specified URI.
    This function ensures that the KeyDB client is ready for asynchronous operations.
    await KeyDBClient.init_session(uri=KEYDB_CACHE_URIS["default_blob"])

@Worker.add_cronjob("*/1 * * * *")
async def test_cron_task(*args, **kwargs):
    logger.info("Cron task ran")
    await asyncio.sleep(5)

@Worker.add_function()
async def test_task(*args, **kwargs):
    logger.info("Task ran")
    await asyncio.sleep(5)