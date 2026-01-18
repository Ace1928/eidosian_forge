from Bio.Application import _Option, _Switch, AbstractCommandline
class ETandemCommandline(_EmbossCommandLine):
    """Commandline object for the etandem program from EMBOSS."""

    def __init__(self, cmd='etandem', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'Sequence', filename=True, is_required=True), _Option(['-minrepeat', 'minrepeat'], 'Minimum repeat size', is_required=True), _Option(['-maxrepeat', 'maxrepeat'], 'Maximum repeat size', is_required=True), _Option(['-threshold', 'threshold'], 'Threshold score'), _Option(['-mismatch', 'mismatch'], 'Allow N as a mismatch'), _Option(['-uniform', 'uniform'], 'Allow uniform consensus'), _Option(['-rformat', 'rformat'], 'Output report format')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)