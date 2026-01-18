from Bio.Application import _Option, _Switch, AbstractCommandline
class DiffseqCommandline(_EmbossCommandLine):
    """Commandline object for the diffseq program from EMBOSS."""

    def __init__(self, cmd='diffseq', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-asequence', 'asequence'], 'First sequence to compare', filename=True, is_required=True), _Option(['-bsequence', 'bsequence'], 'Second sequence to compare', filename=True, is_required=True), _Option(['-wordsize', 'wordsize'], 'Word size to use for comparisons (10 default)', is_required=True), _Option(['-aoutfeat', 'aoutfeat'], "File for output of first sequence's features", filename=True, is_required=True), _Option(['-boutfeat', 'boutfeat'], "File for output of second sequence's features", filename=True, is_required=True), _Option(['-rformat', 'rformat'], 'Output report file format')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)