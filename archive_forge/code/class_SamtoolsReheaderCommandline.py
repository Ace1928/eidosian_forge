from Bio.Application import _Option, _Argument, _Switch
from Bio.Application import AbstractCommandline, _ArgumentList
from Bio.Application import _StaticArgument
class SamtoolsReheaderCommandline(AbstractCommandline):
    """Command line wrapper for samtools reheader.

    Replace the header in in.bam with the header
    in in.header.sam, equivalent to::

    $ samtools reheader <in.header.sam> <in.bam>

    See http://samtools.sourceforge.net/samtools.shtml for more details

    Examples
    --------
    >>> from Bio.Sequencing.Applications import SamtoolsReheaderCommandline
    >>> input_header = "/path/to/header_sam_file"
    >>> input_bam = "/path/to/input_bam_file"
    >>> reheader_cmd = SamtoolsReheaderCommandline(input_header=input_header,
    ...                                            input_bam=input_bam)
    >>> print(reheader_cmd)
    samtools reheader /path/to/header_sam_file /path/to/input_bam_file

    """

    def __init__(self, cmd='samtools', **kwargs):
        """Initialize the class."""
        self.program_name = cmd
        self.parameters = [_StaticArgument('reheader'), _Argument(['input_header', 'header_sam', 'sam_file'], 'Sam file with header', filename=True, is_required=True), _Argument(['input_bam', 'input_file', 'bam_file'], 'BAM file for writing header to', filename=True, is_required=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)