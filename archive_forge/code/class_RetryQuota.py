import threading
class RetryQuota:
    INITIAL_CAPACITY = 500

    def __init__(self, initial_capacity=INITIAL_CAPACITY, lock=None):
        self._max_capacity = initial_capacity
        self._available_capacity = initial_capacity
        if lock is None:
            lock = threading.Lock()
        self._lock = lock

    def acquire(self, capacity_amount):
        """Attempt to aquire a certain amount of capacity.

        If there's not sufficient amount of capacity available, ``False``
        is returned.  Otherwise, ``True`` is returned, which indicates that
        capacity was successfully allocated.

        """
        with self._lock:
            if capacity_amount > self._available_capacity:
                return False
            self._available_capacity -= capacity_amount
            return True

    def release(self, capacity_amount):
        """Release capacity back to the retry quota.

        The capacity being released will be truncated if necessary
        to ensure the max capacity is never exceeded.

        """
        if self._max_capacity == self._available_capacity:
            return
        with self._lock:
            amount = min(self._max_capacity - self._available_capacity, capacity_amount)
            self._available_capacity += amount

    @property
    def available_capacity(self):
        return self._available_capacity