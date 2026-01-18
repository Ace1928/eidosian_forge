from incremental import Version
from ._deprecate import deprecatedGnomeReactor
from twisted.internet import gtk2reactor
class Glib2Reactor(gtk2reactor.Gtk2Reactor):
    """
    The reactor using the glib mainloop.
    """

    def __init__(self):
        """
        Override init to set the C{useGtk} flag.
        """
        gtk2reactor.Gtk2Reactor.__init__(self, useGtk=False)