import binascii
class NotTagError(WrongObjectException):
    """Indicates that the sha requested does not point to a tag."""
    type_name = 'tag'