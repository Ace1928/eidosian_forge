from collections import deque
from typing import Any, Callable, Deque, Dict
Increase/decrease the amount of cached data.

        :param max_size: The maximum number of bytes to cache.
        :param after_cleanup_size: After cleanup, we should have at most this
            many bytes cached. This defaults to 80% of max_size.
        