import Bio.GenBank
class Qualifier:
    """Hold information about a qualifier in a GenBank feature.

    Attributes:
     - key - The key name of the qualifier (ie. /organism=)
     - value - The value of the qualifier ("Dictyostelium discoideum").

    """

    def __init__(self, key='', value=''):
        """Initialize the class."""
        self.key = key
        self.value = value

    def __repr__(self):
        """Representation of the object for debugging or logging."""
        return f'Qualifier(key={self.key!r}, value={self.value!r})'

    def __str__(self):
        """Return feature qualifier as a GenBank format string."""
        output = ' ' * Record.GB_FEATURE_INDENT
        space_wrap = 1
        for no_space_key in Bio.GenBank._BaseGenBankConsumer.remove_space_keys:
            if no_space_key in self.key:
                space_wrap = 0
        return output + _wrapped_genbank(self.key + self.value, Record.GB_FEATURE_INDENT, space_wrap)