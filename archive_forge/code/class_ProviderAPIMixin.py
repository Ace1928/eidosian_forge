class ProviderAPIMixin(object):
    """Allow referencing provider apis on self via __getattr__.

    Be sure this class is first in the class definition for inheritance.
    """

    def __getattr__(self, item):
        """Magic getattr method."""
        try:
            return getattr(ProviderAPIs, item)
        except AttributeError:
            return self.__getattribute__(item)