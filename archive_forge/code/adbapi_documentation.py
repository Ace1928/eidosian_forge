from twisted.internet import threads
from twisted.python import log, reflect

        Disconnect a database connection associated with this pool.

        Note: This function should only be used by the same thread which called
        L{ConnectionPool.connect}. As with C{connect}, this function is not
        used in normal non-threaded Twisted code.
        