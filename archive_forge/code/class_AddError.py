from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddError(Command):
    """
    Add an error.
    """
    arguments = [(b'testName', NativeString()), (b'errorClass', NativeString()), (b'errorStreamId', Integer()), (b'framesStreamId', Integer())]
    response = [(b'success', Boolean())]