from Bio.Application import _Option, _Argument, _Switch, AbstractCommandline
from Bio.Application import _StaticArgument
class BwaIndexCommandline(AbstractCommandline):
    """Command line wrapper for Burrows Wheeler Aligner (BWA) index.

    Index database sequences in the FASTA format, equivalent to::

        $ bwa index [-p prefix] [-a algoType] [-c] <in.db.fasta>

    See http://bio-bwa.sourceforge.net/bwa.shtml for details.

    Examples
    --------
    >>> from Bio.Sequencing.Applications import BwaIndexCommandline
    >>> reference_genome = "/path/to/reference_genome.fasta"
    >>> index_cmd = BwaIndexCommandline(infile=reference_genome, algorithm="bwtsw")
    >>> print(index_cmd)
    bwa index -a bwtsw /path/to/reference_genome.fasta

    You would typically run the command using index_cmd() or via the
    Python subprocess module, as described in the Biopython tutorial.

    """

    def __init__(self, cmd='bwa', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('index'), _Option(['-a', 'a', 'algorithm'], 'Algorithm for constructing BWT index.\n\n                    Available options are:\n                        - is:    IS linear-time algorithm for constructing suffix array.\n                          It requires 5.37N memory where N is the size of the database.\n                          IS is moderately fast, but does not work with database larger\n                          than 2GB. IS is the default algorithm due to its simplicity.\n                        - bwtsw: Algorithm implemented in BWT-SW. This method works with the\n                          whole human genome, but it does not work with database\n                          smaller than 10MB and it is usually slower than IS.', checker_function=lambda x: x in ['is', 'bwtsw'], equate=False, is_required=True), _Option(['-p', 'p', 'prefix'], 'Prefix of the output database [same as db filename]', equate=False, is_required=False), _Argument(['infile'], 'Input file name', filename=True, is_required=True), _Switch(['-c', 'c'], 'Build color-space index. The input fasta should be in nucleotide space.')]
        AbstractCommandline.__init__(self, cmd, **kwargs)