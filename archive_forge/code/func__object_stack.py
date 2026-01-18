import paste.util.threadinglocal as threadinglocal
def _object_stack(self):
    """Returns all of the objects stacked in this container

        (Might return [] if there are none)
        """
    try:
        try:
            objs = self.____local__.objects
        except AttributeError:
            return []
        return objs[:]
    except AssertionError:
        return []