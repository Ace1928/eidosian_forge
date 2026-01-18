import binascii
class NotTreeError(WrongObjectException):
    """Indicates that the sha requested does not point to a tree."""
    type_name = 'tree'