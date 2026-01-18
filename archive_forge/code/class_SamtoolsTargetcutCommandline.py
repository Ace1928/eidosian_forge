from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsTargetcutCommandline(AbstractCommandline):
    """Command line wrapper for samtools targetcut.

    This command identifies target regions by examining the continuity
    of read depth, computes haploid consensus sequences of targets
    and outputs a SAM with each sequence corresponding to a target,
    equivalent to::

        $ samtools targetcut [-Q minBaseQ] [-i inPenalty] [-0 em0]
                             [-1 em1] [-2 em2] [-f ref] <in.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsTargetcutCommandline
    >>> input_bam = "/path/to/aln.bam"
    >>> samtools_targetcut_cmd = SamtoolsTargetcutCommandline(input_bam=input_bam)
    >>> print(samtools_targetcut_cmd)
    samtools targetcut /path/to/aln.bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('targetcut'), _Option(['-Q', 'Q'], 'Minimum Base Quality ', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-i', 'i'], 'Insertion Penalty', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-f', 'f'], 'Reference Filename', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-0', 'em0'], 'em0', equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-1', 'em1'], 'em1', equate=False, checker_function=lambda x: isinstance(x, str)), _Option(['-2', 'em2'], 'em2', equate=False, checker_function=lambda x: isinstance(x, str)), _Argument(['input', 'input_bam', 'in_bam'], 'Input file', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)