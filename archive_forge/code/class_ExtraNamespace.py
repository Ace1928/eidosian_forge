from .compat import escape
from .jsonify import encode
class ExtraNamespace(object):
    """
    Extra variables for the template namespace to pass to the renderer as named
    parameters.

    :param extras: dictionary of extra parameters. Defaults to an empty dict.
    """

    def __init__(self, extras={}):
        self.namespace = dict(extras)

    def update(self, d):
        """
        Updates the extra variable dictionary for the namespace.
        """
        self.namespace.update(d)

    def make_ns(self, ns):
        """
        Returns the `lazily` created template namespace.
        """
        if self.namespace:
            val = {}
            val.update(self.namespace)
            val.update(ns)
            return val
        else:
            return ns