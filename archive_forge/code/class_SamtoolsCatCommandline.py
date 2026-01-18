from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsCatCommandline(AbstractCommandline):
    """Command line wrapper for samtools cat.

    Concatenate BAMs, equivalent to::

        $ samtools cat [-h header.sam] [-o out.bam] <in1.bam> <in2.bam> [ ... ]

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsCatCommandline
    >>> input_bam1 = "/path/to/input_bam1"
    >>> input_bam2 = "/path/to/input_bam2"
    >>> input_bams = [input_bam1, input_bam2]
    >>> samtools_cat_cmd = SamtoolsCatCommandline(input_bam=input_bams)
    >>> print(samtools_cat_cmd)
    samtools cat /path/to/input_bam1 /path/to/input_bam2

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('cat'), _Option(['-h', 'h'], 'Header SAM file', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-o', 'o'], 'Output SAM file', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _ArgumentList(['input', 'input_bam', 'bams'], 'Input BAM files', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)