import warnings
from incremental import Version, getVersionString
from twisted.python.deprecate import DEPRECATION_WARNING_FORMAT

    Emit a deprecation warning about a gnome-related reactor.

    @param name: The name of the reactor.  For example, C{"gtk2reactor"}.

    @param version: The version in which the deprecation was introduced.
    