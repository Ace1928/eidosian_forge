class AutoLinker(object):
    """
    Base class, provides interface for I{automatic} link
    management between a L{Properties} object and the L{Properties}
    contained within I{values}.
    """

    def updated(self, properties, prev, next):
        """
        Notification that a values was updated and the linkage
        between the I{properties} contained with I{prev} need to
        be relinked to the L{Properties} contained within the
        I{next} value.
        """
        pass