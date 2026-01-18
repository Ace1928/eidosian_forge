import collections
from typing import (
import duet.futuretools as futuretools
Allows async iteration over values dynamically added by the client.

    This class is useful for creating an asynchronous iterator that is "fed" by
    one process (the "producer") and iterated over by another process (the
    "consumer"). The producer calls `.add` repeatedly to add values to be
    iterated over, and then calls either `.done` or `.error` to stop the
    iteration or raise an error, respectively. The consumer can use `async for`
    or direct calls to `__anext__` to iterate over the produced values.
    