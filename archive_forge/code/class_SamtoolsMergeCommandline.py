from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsMergeCommandline(AbstractCommandline):
    """Command line wrapper for samtools merge.

    Merge multiple sorted alignments, equivalent to::

        $ samtools merge [-nur1f] [-h inh.sam] [-R reg]
                         <out.bam> <in1.bam> <in2.bam> [...]

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsMergeCommandline
    >>> out_bam = "/path/to/out_bam"
    >>> in_bam = ["/path/to/input_bam1", "/path/to/input_bam2"]
    >>> merge_cmd = SamtoolsMergeCommandline(out_bam=out_bam,
    ...                                      input_bam=in_bam)
    >>> print(merge_cmd)
    samtools merge /path/to/out_bam /path/to/input_bam1 /path/to/input_bam2

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('merge'), _Switch(['-n', 'n'], 'The input alignments are sorted by read names\n                    rather than by chromosomal coordinates'), _Switch(['-r', 'r'], 'Attach an RG tag to each alignment.\n                    The tag value is inferred from file names'), _Switch(['-u', 'u'], 'Uncompressed BAM output'), _Switch(['-1', 'fast_bam'], 'Use zlib compression level 1\n                                           to compress the output'), _Switch(['-f', 'f'], 'Force to overwrite the\n                                    output file if present'), _Option(['-h', 'h'], "Use the lines of FILE as '@'\n                                    headers to be copied to out.bam", filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-R', 'R'], 'Merge files in the specified region indicated by STR', equate=False, checker_function=lambda x: isinstance(x, str)), _Argument(['output_bam', 'out_bam', 'out', 'output'], 'Output BAM file', filename=True, is_required=True), _ArgumentList(['input_bam', 'in_bam', 'input', 'bam'], 'Input BAM', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)