from Bio.Application import _Option, _Switch, AbstractCommandline
class FNeighborCommandline(_EmbossCommandLine):
    """Commandline object for the fneighbor program from EMBOSS.

    fneighbor is an EMBOSS wrapper for the PHYLIP program neighbor used for
    calculating neighbor-joining or UPGMA trees from distance matrices.
    """

    def __init__(self, cmd='fneighbor', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-datafile', 'datafile'], 'dist file to use (phylip)', filename=True, is_required=True), _Option(['-matrixtype', 'matrixtype'], 'is matrix square (S), upper (U) or lower (L)'), _Option(['-treetype', 'treetype'], 'nj or UPGMA tree (n/u)'), _Option(['-outgrno', 'outgrno'], 'taxon to use as OG'), _Option(['-jumble', 'jumble'], 'randommise input order (Y/n)'), _Option(['-seed', 'seed'], 'provide a random seed'), _Option(['-trout', 'trout'], 'write tree (Y/n)'), _Option(['-outtreefile', 'outtreefile'], 'filename for output tree'), _Option(['-progress', 'progress'], 'print progress (Y/n)'), _Option(['-treeprint', 'treeprint'], 'print tree (Y/n)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)