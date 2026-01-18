from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbiblastnCommandline(_NcbiblastMain2SeqCommandline):
    """Wrapper for the NCBI BLAST+ program blastn (for nucleotides).

    With the release of BLAST+ (BLAST rewritten in C++ instead of C), the NCBI
    replaced the old blastall tool with separate tools for each of the searches.
    This wrapper therefore replaces BlastallCommandline with option -p blastn.

    For example, to run a search against the "nt" nucleotide database using the
    FASTA nucleotide file "m_code.fasta" as the query, with an expectation value
    cut off of 0.001, saving the output to a file in XML format:

    >>> from Bio.Blast.Applications import NcbiblastnCommandline
    >>> cline = NcbiblastnCommandline(query="m_cold.fasta", db="nt", strand="plus",
    ...                               evalue=0.001, out="m_cold.xml", outfmt=5)
    >>> cline
    NcbiblastnCommandline(cmd='blastn', out='m_cold.xml', outfmt=5, query='m_cold.fasta', db='nt', evalue=0.001, strand='plus')
    >>> print(cline)
    blastn -out m_cold.xml -outfmt 5 -query m_cold.fasta -db nt -evalue 0.001 -strand plus

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='blastn', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-strand', 'strand'], 'Query strand(s) to search against database/subject.\n\nValues allowed are "both" (default), "minus", "plus".', checker_function=lambda value: value in ['both', 'minus', 'plus'], equate=False), _Option(['-task', 'task'], "Task to execute (string, default 'megablast')\n\nAllowed values 'blastn', 'blastn-short', 'dc-megablast', 'megablast' (the default), or 'vecscreen'.", checker_function=lambda value: value in ['blastn', 'blastn-short', 'dc-megablast', 'megablast', 'vecscreen'], equate=False), _Option(['-penalty', 'penalty'], 'Penalty for a nucleotide mismatch (integer, at most zero).', equate=False), _Option(['-reward', 'reward'], 'Reward for a nucleotide match (integer, at least zero).', equate=False), _Option(['-use_index', 'use_index'], 'Use MegaBLAST database index (Boolean, Default = False)', equate=False), _Option(['-index_name', 'index_name'], 'MegaBLAST database index name.', equate=False), _Option(['-dust', 'dust'], "Filter query sequence with DUST (string).\n\nFormat: 'yes', 'level window linker', or 'no' to disable.\n\nDefault = '20 64 1'.", equate=False), _Option(['-filtering_db', 'filtering_db'], 'BLAST database containing filtering elements (i.e. repeats).', equate=False), _Option(['-window_masker_taxid', 'window_masker_taxid'], 'Enable WindowMasker filtering using a Taxonomic ID (integer).', equate=False), _Option(['-window_masker_db', 'window_masker_db'], 'Enable WindowMasker filtering using this repeats database (string).', equate=False), _Option(['-perc_identity', 'perc_identity'], 'Percent identity (real, 0 to 100 inclusive).', equate=False), _Option(['-template_type', 'template_type'], "Discontiguous MegaBLAST template type (string).\n\nAllowed values: 'coding', 'coding_and_optimal' or 'optimal'.\nRequires: template_length.", checker_function=lambda value: value in ['coding', 'coding_and_optimal', 'optimal'], equate=False), _Option(['-template_length', 'template_length'], 'Discontiguous MegaBLAST template length (integer).\n\nAllowed values: 16, 18, 21.\n\nRequires: template_type.', checker_function=lambda value: value in [16, 18, 21, '16', '18', '21'], equate=False), _Switch(['-no_greedy', 'no_greedy'], 'Use non-greedy dynamic programming extension'), _Option(['-min_raw_gapped_score', 'min_raw_gapped_score'], 'Minimum raw gapped score to keep an alignment in the preliminary gapped and traceback stages (integer).', equate=False), _Switch(['-ungapped', 'ungapped'], 'Perform ungapped alignment only?'), _Option(['-off_diagonal_range', 'off_diagonal_range'], 'Number of off-diagonals to search for the 2nd hit (integer).\n\nExpects a positive integer, or 0 (default) to turn off.Added in BLAST 2.2.23+', equate=False)]
        _NcbiblastMain2SeqCommandline.__init__(self, cmd, **kwargs)

    def _validate(self):
        if self.template_type and (not self.template_length) or (self.template_length and (not self.template_type)):
            raise ValueError('Options template_type and template_type require each other.')
        _NcbiblastMain2SeqCommandline._validate(self)