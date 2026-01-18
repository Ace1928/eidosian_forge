from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbitblastnCommandline(_NcbiblastMain2SeqCommandline):
    """Wrapper for the NCBI BLAST+ program tblastn.

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old blastall tool with separate tools for each of the searches.
    This wrapper therefore replaces BlastallCommandline with option -p tblastn.

    >>> from Bio.Blast.Applications import NcbitblastnCommandline
    >>> cline = NcbitblastnCommandline(help=True)
    >>> cline
    NcbitblastnCommandline(cmd='tblastn', help=True)
    >>> print(cline)
    tblastn -help

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='tblastn', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-task', 'task'], 'Task to execute (string, tblastn (default) or tblastn-fast).', checker_function=lambda value: value in ['tblastn', 'tblastn-fast'], equate=False), _Option(['-db_gencode', 'db_gencode'], 'Genetic code to use to translate query (integer, default 1).', equate=False), _Option(['-frame_shift_penalty', 'frame_shift_penalty'], 'Frame shift penalty (integer, at least 1, default ignored) (OBSOLETE).\n\nThis was removed in BLAST 2.2.27+', equate=False), _Option(['-max_intron_length', 'max_intron_length'], 'Maximum intron length (integer).\n\nLength of the largest intron allowed in a translated nucleotide sequence when linking multiple distinct alignments (a negative value disables linking). Default zero.', equate=False), _Option(['-matrix', 'matrix'], 'Scoring matrix name (default BLOSUM62).', equate=False), _Option(['-threshold', 'threshold'], 'Minimum score for words to be added to the BLAST lookup table (float).', equate=False), _Option(['-comp_based_stats', 'comp_based_stats'], 'Use composition-based statistics (string, default 2, i.e. True).\n\n0, F or f: no composition-based statistics\n\n1: Composition-based statistics as in NAR 29:2994-3005, 2001\n\n2, T or t, D or d : Composition-based score adjustment as in Bioinformatics 21:902-911, 2005, conditioned on sequence properties\n\n3: Composition-based score adjustment as in Bioinformatics 21:902-911, 2005, unconditionally\n\nNote that only tblastn supports values of 1 and 3.', checker_function=lambda value: value in '0Ft12TtDd3', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable.\n\nDefault is "12 2.2 2.5"', equate=False), _Switch(['-ungapped', 'ungapped'], 'Perform ungapped alignment only?'), _Switch(['-use_sw_tback', 'use_sw_tback'], 'Compute locally optimal Smith-Waterman alignments?'), _Option(['-in_pssm', 'in_pssm'], 'PSI-BLAST checkpoint file.\n\nIncompatible with: remote, query', filename=True, equate=False)]
        _NcbiblastMain2SeqCommandline.__init__(self, cmd, **kwargs)