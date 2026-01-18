from Bio.Application import _Option, AbstractCommandline, _Switch
class _Ncbiblast2SeqCommandline(_NcbiblastCommandline):
    """Base Commandline object for (new) NCBI BLAST+ wrappers (PRIVATE).

    This is provided for subclassing, it deals with shared options
    common to all the BLAST tools supporting two-sequence BLAST
    (blastn, psiblast, etc) but not rpsblast or rpstblastn.
    """

    def __init__(self, cmd=None, **kwargs):
        assert cmd is not None
        extra_parameters = [_Option(['-gapopen', 'gapopen'], 'Cost to open a gap (integer).', equate=False), _Option(['-gapextend', 'gapextend'], 'Cost to extend a gap (integer).', equate=False), _Option(['-subject', 'subject'], 'Subject sequence(s) to search.\n\nIncompatible with: db, gilist, seqidlist, negative_gilist, negative_seqidlist, db_soft_mask, db_hard_mask\n\nSee also subject_loc.', filename=True, equate=False), _Option(['-subject_loc', 'subject_loc'], 'Location on the subject sequence (Format: start-stop).\n\nIncompatible with: db, gilist, seqidlist, negative_gilist, negative_seqidlist, db_soft_mask, db_hard_mask, remote.\n\nSee also subject.', equate=False), _Option(['-culling_limit', 'culling_limit'], 'Hit culling limit (integer).\n\nIf the query range of a hit is enveloped by that of at least this many higher-scoring hits, delete the hit.\n\nIncompatible with: best_hit_overhang, best_hit_score_edge.', equate=False), _Option(['-best_hit_overhang', 'best_hit_overhang'], 'Best Hit algorithm overhang value (float, recommended value: 0.1)\n\nFloat between 0.0 and 0.5 inclusive. Incompatible with: culling_limit.', equate=False), _Option(['-best_hit_score_edge', 'best_hit_score_edge'], 'Best Hit algorithm score edge value (float).\n\nFloat between 0.0 and 0.5 inclusive. Recommended value: 0.1\n\nIncompatible with: culling_limit.', equate=False)]
        try:
            self.parameters = extra_parameters + self.parameters
        except AttributeError:
            self.parameters = extra_parameters
        _NcbiblastCommandline.__init__(self, cmd, **kwargs)

    def _validate(self):
        incompatibles = {'subject_loc': ['db', 'gilist', 'negative_gilist', 'seqidlist', 'remote'], 'culling_limit': ['best_hit_overhang', 'best_hit_score_edge'], 'subject': ['db', 'gilist', 'negative_gilist', 'seqidlist']}
        self._validate_incompatibilities(incompatibles)
        _NcbiblastCommandline._validate(self)