from Bio.Application import _Option, _Switch, AbstractCommandline
class FProtDistCommandline(_EmbossCommandLine):
    """Commandline object for the fprotdist program from EMBOSS.

    fprotdist is an EMBOSS wrapper for the PHYLIP program protdist used to
    estimate trees from protein sequences using parsimony
    """

    def __init__(self, cmd='fprotdist', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-sequence', 'sequence'], 'seq file to use (phylip)', filename=True, is_required=True), _Option(['-ncategories', 'ncategories'], 'number of rate catergories (1-9)'), _Option(['-rate', 'rate'], 'rate for each category'), _Option(['-catergories', 'catergories'], 'file of rates'), _Option(['-weights', 'weights'], 'weights file'), _Option(['-method', 'method'], 'sub. model [j,h,d,k,s,c]'), _Option(['-gamma', 'gamma'], 'gamma [g, i,c]'), _Option(['-gammacoefficient', 'gammacoefficient'], 'value for gamma (> 0.001)'), _Option(['-invarcoefficient', 'invarcoefficient'], 'float for variation of substitution rate among sites'), _Option(['-aacateg', 'aacateg'], 'Choose the category to use [G,C,H]'), _Option(['-whichcode', 'whichcode'], 'genetic code [c,m,v,f,y]'), _Option(['-ease', 'ease'], 'Pob change catergory (float between -0 and 1)'), _Option(['-ttratio', 'ttratio'], 'Transition/transversion ratio (0-1)'), _Option(['-basefreq', 'basefreq'], 'DNA base frequencies (space separated list)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)