from ..common.utils import struct_parse, dwarf_assert
class AbbrevDecl(object):
    """ Wraps a parsed abbreviation declaration, exposing its fields with
        dict-like access, and adding some convenience methods.

        The abbreviation declaration represents an "entry" that points to it.
    """

    def __init__(self, code, decl):
        self.code = code
        self.decl = decl

    def has_children(self):
        """ Does the entry have children?
        """
        return self['children_flag'] == 'DW_CHILDREN_yes'

    def iter_attr_specs(self):
        """ Iterate over the attribute specifications for the entry. Yield
            (name, form) pairs.
        """
        for attr_spec in self['attr_spec']:
            yield (attr_spec.name, attr_spec.form)

    def __getitem__(self, entry):
        return self.decl[entry]