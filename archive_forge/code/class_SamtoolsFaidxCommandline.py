from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsFaidxCommandline(AbstractCommandline):
    """Command line wrapper for samtools faidx.

    Retrieve and print stats in the index file, equivalent to::

    $ samtools faidx <ref.fasta> [region1 [...]]

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsFaidxCommandline
    >>> reference = "/path/to/reference.fasta"
    >>> samtools_faidx_cmd = SamtoolsFaidxCommandline(reference=reference)
    >>> print(samtools_faidx_cmd)
    samtools faidx /path/to/reference.fasta

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('faidx'), _Argument(['reference', 'reference_fasta', 'ref'], 'Reference FASTA to be indexed', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)