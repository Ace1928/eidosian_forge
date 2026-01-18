from typing import Union
from gi.repository import GLib
from twisted.internet import _glibbase
from twisted.internet.error import ReactorAlreadyRunning
from twisted.python import runtime
class PortableGIReactor(_glibbase.GlibReactorBase):
    """
    Portable GObject Introspection event loop reactor.
    """

    def __init__(self, useGtk=False):
        super().__init__(GLib, None, useGtk=useGtk)

    def registerGApplication(self, app):
        """
        Register a C{Gio.Application} or C{Gtk.Application}, whose main loop
        will be used instead of the default one.
        """
        raise NotImplementedError('GApplication is not currently supported on Windows.')

    def simulate(self) -> None:
        """
        For compatibility only. Do nothing.
        """