from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsPhaseCommandline(AbstractCommandline):
    """Command line wrapper for samtools phase.

    Call and phase heterozygous SNPs, equivalent to::

        $ samtools phase [-AF] [-k len] [-b prefix]
                         [-q minLOD] [-Q minBaseQ] <in.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsPhaseCommandline
    >>> input_bam = "/path/to/in.bam"
    >>> samtools_phase_cmd = SamtoolsPhaseCommandline(input_bam=input_bam)
    >>> print(samtools_phase_cmd)
    samtools phase /path/to/in.bam

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('phase'), _Argument(['input', 'input_bam', 'in_bam'], 'Input file', filename=True, is_required=True), _Switch(['-A', 'A'], 'Drop reads with ambiguous phase'), _Option(['-b', 'b'], 'Prefix of BAM output', filename=True, equate=False, checker_function=lambda x: isinstance(x, str)), _Switch(['-F', 'F'], 'Do not attempt to fix chimeric reads'), _Option(['-k', 'k'], 'Maximum length for local phasing', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-q', 'q'], 'Minimum Phred-scaled LOD to\n                    call a heterozygote', equate=False, checker_function=lambda x: isinstance(x, int)), _Option(['-Q', 'Q'], 'Minimum base quality to be\n                    used in het calling', equate=False, checker_function=lambda x: isinstance(x, int))]
        AbstractCommandline.__init__(self, cmd, **kwargs)