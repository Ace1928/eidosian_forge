from . import errors
Acquire the lock in write mode.

        If the lock was originally acquired in read mode this will fail.

        :param token: If given and the lock is already held,
            then validate that we already hold the real
            lock with this token.

        :returns: The token from the underlying lock.
        