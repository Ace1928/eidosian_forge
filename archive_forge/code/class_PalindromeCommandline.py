from Bio.Application import _Option, _Switch, AbstractCommandline
class PalindromeCommandline(_EmbossCommandLine):
    """Commandline object for the palindrome program from EMBOSS."""

    def __init__(self, cmd='palindrome', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Sequence', filename=True, is_required=True), _Option(['-minpallen', 'minpallen'], 'Minimum palindrome length', is_required=True), _Option(['-maxpallen', 'maxpallen'], 'Maximum palindrome length', is_required=True), _Option(['-gaplimit', 'gaplimit'], 'Maximum gap between repeats', is_required=True), _Option(['-nummismatches', 'nummismatches'], 'Number of mismatches allowed', is_required=True), _Option(['-overlap', 'overlap'], 'Report overlapping matches', is_required=True)]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)