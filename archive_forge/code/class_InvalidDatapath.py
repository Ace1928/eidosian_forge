from os_ken import exception
class InvalidDatapath(_ExceptionBase):
    """Datapath is invalid.

    This can happen when the bridge disconnects.
    """
    message = 'Datapath Invalid %(result)s'