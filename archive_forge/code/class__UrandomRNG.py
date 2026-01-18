from os import urandom
class _UrandomRNG(object):

    def read(self, n):
        """Return a random byte string of the desired size."""
        return urandom(n)

    def flush(self):
        """Method provided for backward compatibility only."""
        pass

    def reinit(self):
        """Method provided for backward compatibility only."""
        pass

    def close(self):
        """Method provided for backward compatibility only."""
        pass