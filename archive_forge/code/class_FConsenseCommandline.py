from Bio.Application import _Option, _Switch, AbstractCommandline
class FConsenseCommandline(_EmbossCommandLine):
    """Commandline object for the fconsense program from EMBOSS.

    fconsense is an EMBOSS wrapper for the PHYLIP program consense used to
    calculate consensus trees.
    """

    def __init__(self, cmd='fconsense', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-intreefile', 'intreefile'], 'file with phylip trees to make consensus from', filename=True, is_required=True), _Option(['-method', 'method'], 'consensus method [s, mr, MRE, ml]'), _Option(['-mlfrac', 'mlfrac'], 'cut-off freq for branch to appear in consensus (0.5-1.0)'), _Option(['-root', 'root'], 'treat trees as rooted (YES, no)'), _Option(['-outgrno', 'outgrno'], 'OTU to use as outgroup (starts from 0)'), _Option(['-trout', 'trout'], 'treat trees as rooted (YES, no)'), _Option(['-outtreefile', 'outtreefile'], 'Phylip tree output file (optional)')]
        _EmbossCommandLine.__init__(self, cmd, **kwargs)