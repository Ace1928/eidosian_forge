from Bio.Application import _Option, _Argument, _Switch, AbstractCommandline
from Bio.Application import _StaticArgument
class BwaBwaswCommandline(AbstractCommandline):
    """Command line wrapper for Burrows Wheeler Aligner (BWA) bwasw.

    Align query sequences from FASTQ files. Equivalent to::

        $ bwa bwasw [...] <in.db.fasta> <in.fq>

    See http://bio-bwa.sourceforge.net/bwa.shtml for details.

    Examples
    --------
    >>> from Bio.Sequencing.Applications import BwaBwaswCommandline
    >>> reference_genome = "/path/to/reference_genome.fasta"
    >>> read_file = "/path/to/read_1.fq"
    >>> bwasw_cmd = BwaBwaswCommandline(reference=reference_genome, read_file=read_file)
    >>> print(bwasw_cmd)
    bwa bwasw /path/to/reference_genome.fasta /path/to/read_1.fq

    You would typically run the command line using bwasw_cmd() or via the
    Python subprocess module, as described in the Biopython tutorial.

    """

    def __init__(self, cmd='bwa', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('bwasw'), _Argument(['reference'], 'Reference file name', filename=True, is_required=True), _Argument(['read_file'], 'Read file', filename=True, is_required=True), _Argument(['mate_file'], 'Mate file', filename=True, is_required=False), _Option(['-a', 'a'], 'Score of a match [1]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-b', 'b'], 'Mismatch penalty [3]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-q', 'q'], 'Gap open penalty [5]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-r', 'r'], 'Gap extension penalty. The penalty for a contiguous gap of size k is q+k*r. [2]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-t', 't'], 'Number of threads in the multi-threading mode [1]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-w', 'w'], 'Band width in the banded alignment [33]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-T', 'T'], 'Minimum score threshold divided by a [37]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-c', 'c'], 'Coefficient for threshold adjustment according to query length [5.5].\n\n                    Given an l-long query, the threshold for a hit to be retained is\n                    a*max{T,c*log(l)}.', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['-z', 'z'], 'Z-best heuristics. Higher -z increases accuracy at the cost of speed. [1]', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-s', 's'], 'Maximum SA interval size for initiating a seed [3].\n\n                    Higher -s increases accuracy at the cost of speed.', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['-N', 'N'], 'Minimum number of seeds supporting the resultant alignment to skip reverse alignment. [5]', checker_function=lambda x: isinstance(x, int), equate=False)]
        AbstractCommandline.__init__(self, cmd, **kwargs)