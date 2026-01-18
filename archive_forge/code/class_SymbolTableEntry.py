from pyparsing import *
from sys import stdin, argv, exit
class SymbolTableEntry(object):
    """Class which represents one symbol table entry."""

    def __init__(self, sname='', skind=0, stype=0, sattr=None, sattr_name='None'):
        """Initialization of symbol table entry.
           sname - symbol name
           skind - symbol kind
           stype - symbol type
           sattr - symbol attribute
           sattr_name - symbol attribute name (used only for table display)
        """
        self.name = sname
        self.kind = skind
        self.type = stype
        self.attribute = sattr
        self.attribute_name = sattr_name
        self.param_types = []

    def set_attribute(self, name, value):
        """Sets attribute's name and value"""
        self.attribute_name = name
        self.attribute = value

    def attribute_str(self):
        """Returns attribute string (used only for table display)"""
        return '{0}={1}'.format(self.attribute_name, self.attribute) if self.attribute != None else 'None'