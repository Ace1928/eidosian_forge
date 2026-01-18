import logging
from botocore import xform_name
class Identifier(object):
    """
    A resource identifier, given by its name.

    :type name: string
    :param name: The name of the identifier
    """

    def __init__(self, name, member_name=None):
        self.name = name
        self.member_name = member_name