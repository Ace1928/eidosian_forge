from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsVersion0xSortCommandline(AbstractCommandline):
    """Command line wrapper for samtools version 0.1.x sort.

    Concatenate BAMs, equivalent to::

    $ samtools sort [-no] [-m maxMem] <in.bam> <out.prefix>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsVersion0xSortCommandline
    >>> input_bam = "/path/to/input_bam"
    >>> out_prefix = "/path/to/out_prefix"
    >>> samtools_sort_cmd = SamtoolsVersion0xSortCommandline(input=input_bam, out_prefix=out_prefix)
    >>> print(samtools_sort_cmd)
    samtools sort /path/to/input_bam /path/to/out_prefix

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('sort'), _Switch(['-o', 'o'], 'Output the final alignment\n                                    to the standard output'), _Switch(['-n', 'n'], 'Sort by read names rather\n                                    than by chromosomal coordinates'), _Option(['-m', 'm'], 'Approximately the maximum required memory', equate=False, checker_function=lambda x: isinstance(x, int)), _Argument(['input'], 'Input BAM file', filename=True, is_required=True), _Argument(['out_prefix'], 'Output prefix', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)