from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class TestWrite(Command):
    """
    Write test log.
    """
    arguments = [(b'out', NativeString())]
    response = [(b'success', Boolean())]