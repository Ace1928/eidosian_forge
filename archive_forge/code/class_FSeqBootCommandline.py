from Bio.Application import _Option, _Switch, AbstractCommandline
class FSeqBootCommandline(_EmbossCommandLine):
    """Commandline object for the fseqboot program from EMBOSS.

    fseqboot is an EMBOSS wrapper for the PHYLIP program seqboot used to
    pseudo-sample alignment files.
    """

    def __init__(self, cmd='fseqboot', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'seq file to sample (phylip)', filename=True, is_required=True), _Option(['-categories', 'catergories'], 'file of input categories'), _Option(['-weights', 'weights'], ' weights file'), _Option(['-test', 'test'], 'specify operation, default is bootstrap'), _Option(['-regular', 'regular'], 'absolute number to resample'), _Option(['-fracsample', 'fracsample'], 'fraction to resample'), _Option(['-rewriteformat', 'rewriteformat'], 'output format ([P]hyilp, [n]exus, [x]ml'), _Option(['-seqtype', 'seqtype'], 'output format ([D]na, [p]rotein, [r]na'), _Option(['-blocksize', 'blocksize'], 'print progress (Y/n)'), _Option(['-reps', 'reps'], 'how many replicates, defaults to 100)'), _Option(['-justweights', 'jusweights'], 'what to write out [D]atasets of just [w]eights'), _Option(['-seed', 'seed'], 'specify random seed'), _Option(['-dotdiff', 'dotdiff'], 'Use dot-differencing? [Y/n]')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)