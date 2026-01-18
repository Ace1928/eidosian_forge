from Bio.Application import _Option, _Switch, AbstractCommandline
class _EmbossMinimalCommandLine(AbstractCommandline):
    """Base Commandline object for EMBOSS wrappers (PRIVATE).

    This is provided for subclassing, it deals with shared options
    common to all the EMBOSS tools:

    Attributes:
     - auto               Turn off prompts
     - stdout             Write standard output
     - filter             Read standard input, write standard output
     - options            Prompt for standard and additional values
     - debug              Write debug output to program.dbg
     - verbose            Report some/full command line options
     - help               Report command line options. More
                          information on associated and general
                          qualifiers can be found with -help -verbose
     - warning            Report warnings
     - error              Report errors
     - fatal              Report fatal errors
     - die                Report dying program messages

    """

    def __init__(self, cmd=None, **kwargs):
        assert cmd is not None
        extra_parameters = [_Switch(['-auto', 'auto'], 'Turn off prompts.\n\nAutomatic mode disables prompting, so we recommend you set this argument all the time when calling an EMBOSS tool from Biopython.'), _Switch(['-stdout', 'stdout'], 'Write standard output.'), _Switch(['-filter', 'filter'], 'Read standard input, write standard output.'), _Switch(['-options', 'options'], 'Prompt for standard and additional values.\n\nIf you are calling an EMBOSS tool from within Biopython, we DO NOT recommend using this option.'), _Switch(['-debug', 'debug'], 'Write debug output to program.dbg.'), _Switch(['-verbose', 'verbose'], 'Report some/full command line options'), _Switch(['-help', 'help'], 'Report command line options.\n\nMore information on associated and general qualifiers can be found with -help -verbose'), _Switch(['-warning', 'warning'], 'Report warnings.'), _Switch(['-error', 'error'], 'Report errors.'), _Switch(['-die', 'die'], 'Report dying program messages.')]
        try:
            self.parameters = extra_parameters + self.parameters
        except AttributeError:
            self.parameters = extra_parameters
        AbstractCommandline.__init__(self, cmd, **kwargs)