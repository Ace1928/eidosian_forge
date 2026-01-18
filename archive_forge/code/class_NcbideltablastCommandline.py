from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbideltablastCommandline(_Ncbiblast2SeqCommandline):
    """Create a commandline for the NCBI BLAST+ program deltablast (for proteins).

    This is a wrapper for the deltablast command line command included in
    the NCBI BLAST+ software (not present in the original BLAST).

    >>> from Bio.Blast.Applications import NcbideltablastCommandline
    >>> cline = NcbideltablastCommandline(query="rosemary.pro", db="nr",
    ...                               evalue=0.001, remote=True)
    >>> cline
    NcbideltablastCommandline(cmd='deltablast', query='rosemary.pro', db='nr', evalue=0.001, remote=True)
    >>> print(cline)
    deltablast -query rosemary.pro -db nr -evalue 0.001 -remote

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='deltablast', **kwargs):
        """Initialize the class."""
        self.parameters = [_Option(['-matrix', 'matrix'], 'Scoring matrix name (default BLOSUM62).'), _Option(['-threshold', 'threshold'], 'Minimum score for words to be added to the BLAST lookup table (float).', equate=False), _Option(['-comp_based_stats', 'comp_based_stats'], 'Use composition-based statistics (string, default 2, i.e. True).\n\n0, F or f: no composition-based statistics.\n\n2, T or t, D or d : Composition-based score adjustment as in Bioinformatics 21:902-911, 2005, conditioned on sequence properties\n\nNote that tblastn also supports values of 1 and 3.', checker_function=lambda value: value in '0Ft2TtDd', equate=False), _Option(['-seg', 'seg'], 'Filter query sequence with SEG (string).\n\nFormat: "yes", "window locut hicut", or "no" to disable. Default is "12 2.2 2.5"', equate=False), _Option(['-gap_trigger', 'gap_trigger'], 'Number of bits to trigger gapping. Default = 22.', equate=False), _Switch(['-use_sw_tback', 'use_sw_tback'], 'Compute locally optimal Smith-Waterman alignments?'), _Option(['-num_iterations', 'num_iterations'], 'Number of iterations to perform. (integer >=1, Default is 1).\n\nIncompatible with: remote', equate=False), _Option(['-out_pssm', 'out_pssm'], 'File name to store checkpoint file.', filename=True, equate=False), _Option(['-out_ascii_pssm', 'out_ascii_pssm'], 'File name to store ASCII version of PSSM.', filename=True, equate=False), _Switch(['-save_pssm_after_last_round', 'save_pssm_after_last_round'], 'Save PSSM after the last database search.'), _Switch(['-save_each_pssm', 'save_each_pssm'], 'Save PSSM after each iteration.\n\nFile name is given in -save_pssm or -save_ascii_pssm options.'), _Option(['-pseudocount', 'pseudocount'], 'Pseudo-count value used when constructing PSSM (integer, default 0).', equate=False), _Option(['-domain_inclusion_ethresh', 'domain_inclusion_ethresh'], 'E-value inclusion threshold for alignments with conserved domains.\n\n(float, Default is 0.05)', equate=False), _Option(['-inclusion_ethresh', 'inclusion_ethresh'], 'Pairwise alignment e-value inclusion threshold (float, default 0.002).', equate=False), _Option(['-rpsdb', 'rpsdb'], "BLAST domain database name (dtring, Default = 'cdd_delta').", equate=False), _Switch(['-show_domain_hits', 'show_domain_hits'], 'Show domain hits?\n\nIncompatible with: remote, subject')]
        _Ncbiblast2SeqCommandline.__init__(self, cmd, **kwargs)