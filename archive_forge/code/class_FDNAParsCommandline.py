from Bio.Application import _Option, _Switch, AbstractCommandline
class FDNAParsCommandline(_EmbossCommandLine):
    """Commandline object for the fdnapars program from EMBOSS.

    fdnapars is an EMBOSS version of the PHYLIP program dnapars, for
    estimating trees from DNA sequences using parsiomny. Calling this command
    without providing a value for the option "-intreefile" will invoke
    "interactive mode" (and as a result fail if called with subprocess) if
    "-auto" is not set to true.
    """

    def __init__(self, cmd='fdnapars', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'seq file to use (phylip)', filename=True, is_required=True), _Option(['-intreefile', 'intreefile'], 'Phylip tree file'), _Option(['-weights', 'weights'], 'weights file'), _Option(['-maxtrees', 'maxtrees'], 'max trees to save during run'), _Option(['-thorough', 'thorough'], 'more thorough search (Y/n)'), _Option(['-rearrange', 'rearrange'], 'Rearrange on just 1 best tree (Y/n)'), _Option(['-transversion', 'transversion'], 'Use tranversion parsimony (y/N)'), _Option(['-njumble', 'njumble'], 'number of times to randomise input order (default is 0)'), _Option(['-seed', 'seed'], 'provide random seed'), _Option(['-outgrno', 'outgrno'], 'Specify outgroup'), _Option(['-thresh', 'thresh'], 'Use threshold parsimony (y/N)'), _Option(['-threshold', 'threshold'], 'Threshold value'), _Option(['-trout', 'trout'], 'Write trees to file (Y/n)'), _Option(['-outtreefile', 'outtreefile'], 'filename for output tree'), _Option(['-dotdiff', 'dotdiff'], 'Use dot-differencing? [Y/n]')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)