from zaqarclient.queues.v1 import core
Ensures pool exists

        This method is not race safe,
        the pool could've been deleted
        right after it was called.
        