import paste.util.threadinglocal as threadinglocal
def _pop_object(self, obj=None):
    """Remove a thread-local object.

        If ``obj`` is given, it is checked against the popped object and an
        error is emitted if they don't match.

        """
    try:
        popped = self.____local__.objects.pop()
        if obj and popped is not obj:
            raise AssertionError('The object popped (%s) is not the same as the object expected (%s)' % (popped, obj))
    except AttributeError:
        raise AssertionError('No object has been registered for this thread')