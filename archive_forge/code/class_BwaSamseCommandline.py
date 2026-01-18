from Bio.Application import _Option, _Argument, _Switch, AbstractCommandline
from Bio.Application import _StaticArgument
class BwaSamseCommandline(AbstractCommandline):
    """Command line wrapper for Burrows Wheeler Aligner (BWA) samse.

    Generate alignments in the SAM format given single-end reads.
    Equvialent to::

        $ bwa samse [-n maxOcc] <in.db.fasta> <in.sai> <in.fq> > <out.sam>

    See http://bio-bwa.sourceforge.net/bwa.shtml for details.

    Examples
    --------
    >>> from Bio.Sequencing.Applications import BwaSamseCommandline
    >>> reference_genome = "/path/to/reference_genome.fasta"
    >>> read_file = "/path/to/read_1.fq"
    >>> sai_file = "/path/to/read_1.sai"
    >>> output_sam_file = "/path/to/read_1.sam"
    >>> samse_cmd = BwaSamseCommandline(reference=reference_genome,
    ...                                 read_file=read_file, sai_file=sai_file)
    >>> print(samse_cmd)
    bwa samse /path/to/reference_genome.fasta /path/to/read_1.sai /path/to/read_1.fq

    You would typically run the command line using samse_cmd(stdout=output_sam_file)
    or via the Python subprocess module, as described in the Biopython tutorial.

    """

    def __init__(self, cmd='bwa', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('samse'), _Argument(['reference'], 'Reference file name', filename=True, is_required=True), _Argument(['sai_file'], 'Sai file name', filename=True, is_required=True), _Argument(['read_file'], 'Read  file name', filename=True, is_required=True), _Option(['-n', 'n'], 'Maximum number of alignments to output in the XA tag for reads paired properly.\n\n                    If a read has more than INT hits, the XA tag will not be written. [3]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-r', 'r'], "Specify the read group in a format like '@RG\tID:foo\tSM:bar'. [null]", checker_function=lambda x: isinstance(x, int), equate=False)]
        AbstractCommandline.__init__(self, cmd, **kwargs)