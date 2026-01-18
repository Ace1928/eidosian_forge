from Bio.Application import _Option, _Switch, AbstractCommandline
class FuzzproCommandline(_EmbossCommandLine):
    """Commandline object for the fuzzpro program from EMBOSS."""

    def __init__(self, cmd='fuzzpro', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Sequence database USA', is_required=True), _Option(['-pattern', 'pattern'], 'Search pattern, using standard IUPAC one-letter codes', is_required=True), _Option(['-pmismatch', 'pmismatch'], 'Number of mismatches'), _Option(['-rformat', 'rformat'], 'Specify the report format to output in.')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)