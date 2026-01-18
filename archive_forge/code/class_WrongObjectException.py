import binascii
class WrongObjectException(Exception):
    """Baseclass for all the _ is not a _ exceptions on objects.

    Do not instantiate directly.

    Subclasses should define a type_name attribute that indicates what
    was expected if they were raised.
    """
    type_name: str

    def __init__(self, sha, *args, **kwargs) -> None:
        Exception.__init__(self, f'{sha} is not a {self.type_name}')