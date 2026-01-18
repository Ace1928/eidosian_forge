from . import _gi
class BoundSignal(str):
    """
        Temporary binding object which can be used for connecting signals
        without specifying the signal name string to connect.
        """

    def __new__(cls, name, *args, **kargs):
        return str.__new__(cls, name)

    def __init__(self, signal, gobj):
        str.__init__(self)
        self.signal = signal
        self.gobj = gobj

    def __repr__(self):
        return 'BoundSignal("%s")' % self

    def __call__(self, *args, **kargs):
        """Call the signals closure."""
        return self.signal.func(self.gobj, *args, **kargs)

    def connect(self, callback, *args, **kargs):
        """Same as GObject.Object.connect except there is no need to specify
            the signal name."""
        return self.gobj.connect(self, callback, *args, **kargs)

    def connect_detailed(self, callback, detail, *args, **kargs):
        """Same as GObject.Object.connect except there is no need to specify
            the signal name. In addition concats "::<detail>" to the signal name
            when connecting; for use with notifications like "notify" when a property
            changes.
            """
        return self.gobj.connect(self + '::' + detail, callback, *args, **kargs)

    def disconnect(self, handler_id):
        """Same as GObject.Object.disconnect."""
        self.gobj.disconnect(handler_id)

    def emit(self, *args, **kargs):
        """Same as GObject.Object.emit except there is no need to specify
            the signal name."""
        return self.gobj.emit(str(self), *args, **kargs)