import queue
import sys
import threading
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union

        Runs the code in the work queue until a result is available from the future.
        Should be run from the thread the executor is initialised in.
        