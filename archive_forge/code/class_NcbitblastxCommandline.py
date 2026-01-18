from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbitblastxCommandline(_NcbiblastMain2SeqCommandline):
    """Wrapper for the NCBI BLAST+ program tblastx.

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old blastall tool with separate tools for each of the searches.
    This wrapper therefore replaces BlastallCommandline with option -p tblastx.

    >>> from Bio.Blast.Applications import NcbitblastxCommandline
    >>> cline = NcbitblastxCommandline(help=True)
    >>> cline
    NcbitblastxCommandline(cmd='tblastx', help=True)
    >>> print(cline)
    tblastx -help

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='tblastx', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-strand', 'strand'], 'Query strand(s) to search against database/subject.\n\nValues allowed are "both" (default), "minus", "plus".', checker_function=lambda value: value in ['both', 'minus', 'plus'], equate=False), _Option(['-query_gencode', 'query_gencode'], 'Genetic code to use to translate query (integer, default 1).', equate=False), _Option(['-db_gencode', 'db_gencode'], 'Genetic code to use to translate query (integer, default 1).', equate=False), _Option(['-max_intron_length', 'max_intron_length'], 'Maximum intron length (integer).\n\nLength of the largest intron allowed in a translated nucleotide sequence when linking multiple distinct alignments (a negative value disables linking). Default zero.', equate=False), _Option(['-matrix', 'matrix'], 'Scoring matrix name (default BLOSUM62).', equate=False), _Option(['-threshold', 'threshold'], 'Minimum score for words to be added to the BLAST lookup table (float).', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable.\n\nDefault is "12 2.2 2.5"', equate=False)]
        _NcbiblastMain2SeqCommandline.__init__(self, cmd, **kwargs)