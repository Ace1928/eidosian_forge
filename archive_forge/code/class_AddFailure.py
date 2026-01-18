from twisted.protocols.amp import Boolean, Command, Integer, Unicode
class AddFailure(Command):
    """
    Add a failure.
    """
    arguments = [(b'testName', NativeString()), (b'failStreamId', Integer()), (b'failClass', NativeString()), (b'framesStreamId', Integer())]
    response = [(b'success', Boolean())]