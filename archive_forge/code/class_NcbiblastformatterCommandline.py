from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbiblastformatterCommandline(_NcbibaseblastCommandline):
    """Wrapper for the NCBI BLAST+ program blast_formatter.

    With the release of BLAST 2.2.24+ (i.e. the BLAST suite rewritten in C++
    instead of C), the NCBI added the ASN.1 output format option to all the
    search tools, and extended the blast_formatter to support this as input.

    The blast_formatter command allows you to convert the ASN.1 output into
    the other output formats (XML, tabular, plain text, HTML).

    >>> from Bio.Blast.Applications import NcbiblastformatterCommandline
    >>> cline = NcbiblastformatterCommandline(archive="example.asn", outfmt=5, out="example.xml")
    >>> cline
    NcbiblastformatterCommandline(cmd='blast_formatter', out='example.xml', outfmt=5, archive='example.asn')
    >>> print(cline)
    blast_formatter -out example.xml -outfmt 5 -archive example.asn

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.

    Note that this wrapper is for the version of blast_formatter from BLAST
    2.2.24+ (or later) which is when the NCBI first announced the inclusion
    this tool. There was actually an early version in BLAST 2.2.23+ (and
    possibly in older releases) but this did not have the -archive option
    (instead -rid is a mandatory argument), and is not supported by this
    wrapper.
    """

    def __init__(self, cmd='blast_formatter', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-rid', 'rid'], 'BLAST Request ID (RID), not compatible with archive arg.', equate=False), _Option(['-archive', 'archive'], 'Archive file of results, not compatible with rid arg.', filename=True, equate=False), _Option(['-max_target_seqs', 'max_target_seqs'], 'Maximum number of aligned sequences to keep.', checker_function=lambda value: value >= 1, equate=False)]
        _NcbibaseblastCommandline.__init__(self, cmd, **kwargs)

    def _validate(self):
        incompatibles = {'rid': ['archive']}
        self._validate_incompatibilities(incompatibles)
        _NcbibaseblastCommandline._validate(self)