class AllocatorMemoryException(Exception):
    """The buffer is not large enough to fulfil an allocation.

    Raised by `Allocator` methods when the operation failed due to
    lack of buffer space.  The buffer should be increased to at least
    requested_capacity and then the operation retried (guaranteed to
    pass second time).
    """

    def __init__(self, requested_capacity):
        self.requested_capacity = requested_capacity