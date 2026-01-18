from Bio.Application import _Option, _Switch, AbstractCommandline
class FuzznucCommandline(_EmbossCommandLine):
    """Commandline object for the fuzznuc program from EMBOSS."""

    def __init__(self, cmd='fuzznuc', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Sequence database USA', is_required=True), _Option(['-pattern', 'pattern'], 'Search pattern, using standard IUPAC one-letter codes', is_required=True), _Option(['-pmismatch', 'pmismatch'], 'Number of mismatches'), _Option(['-complement', 'complement'], 'Search complementary strand'), _Option(['-rformat', 'rformat'], 'Specify the report format to output in.')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)