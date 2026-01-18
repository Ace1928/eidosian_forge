from Bio.Application import _Option, _Switch, AbstractCommandline
class EInvertedCommandline(_EmbossCommandLine):
    """Commandline object for the einverted program from EMBOSS."""

    def __init__(self, cmd='einverted', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Sequence', filename=True, is_required=True), _Option(['-gap', 'gap'], 'Gap penalty', filename=True, is_required=True), _Option(['-threshold', 'threshold'], 'Minimum score threshold', is_required=True), _Option(['-match', 'match'], 'Match score', is_required=True), _Option(['-mismatch', 'mismatch'], 'Mismatch score', is_required=True), _Option(['-maxrepeat', 'maxrepeat'], 'Maximum separation between the start and end of repeat')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)