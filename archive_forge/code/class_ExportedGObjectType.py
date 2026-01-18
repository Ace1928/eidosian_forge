from gi.repository import GObject
import dbus.service
class ExportedGObjectType(GObject.GObject.__class__, dbus.service.InterfaceType):
    """A metaclass which inherits from both GObjectMeta and
    `dbus.service.InterfaceType`. Used as the metaclass for `ExportedGObject`.
    """

    def __init__(cls, name, bases, dct):
        GObject.GObject.__class__.__init__(cls, name, bases, dct)
        dbus.service.InterfaceType.__init__(cls, name, bases, dct)