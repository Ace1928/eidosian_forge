from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddSkip(Command):
    """
    Add a skip.
    """
    arguments = [(b'testName', NativeString()), (b'reason', NativeString())]
    response = [(b'success', Boolean())]