from Bio.Application import _Option, _Switch, AbstractCommandline
class _EmbossCommandLine(_EmbossMinimalCommandLine):
    """Base Commandline object for EMBOSS wrappers (PRIVATE).

    This is provided for subclassing, it deals with shared options
    common to all the EMBOSS tools plus:

     - outfile            Output filename

    """

    def __init__(self, cmd=None, **kwargs):
        assert cmd is not None
        extra_parameters = [_Option(['-outfile', 'outfile'], 'Output filename', filename=True)]
        try:
            self.parameters = extra_parameters + self.parameters
        except AttributeError:
            self.parameters = extra_parameters
        _EmbossMinimalCommandLine.__init__(self, cmd, **kwargs)

    def _validate(self):
        if not (self.outfile or self.filter or self.stdout):
            raise ValueError('You must either set outfile (output filename), or enable filter or stdout (output to stdout).')
        return _EmbossMinimalCommandLine._validate(self)