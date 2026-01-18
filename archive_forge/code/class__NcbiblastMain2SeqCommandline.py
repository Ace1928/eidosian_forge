from Bio.Application import _Option, AbstractCommandline, _Switch
class _NcbiblastMain2SeqCommandline(_Ncbiblast2SeqCommandline):
    """Base Commandline object for (new) NCBI BLAST+ wrappers (PRIVATE).

    This is provided for subclassing, it deals with shared options
    common to the main BLAST tools blastp, blastn, blastx, tblastx, tblastn
    but not psiblast, rpsblast or rpstblastn.
    """

    def __init__(self, cmd=None, **kwargs):
        assert cmd is not None
        extra_parameters = [_Option(['-db_soft_mask', 'db_soft_mask'], 'Filtering algorithm for soft masking (integer).\n\nFiltering algorithm ID to apply to BLAST database as soft masking. Incompatible with: db_hard_mask, subject, subject_loc', equate=False), _Option(['-db_hard_mask', 'db_hard_mask'], 'Filtering algorithm for hard masking (integer).\n\nFiltering algorithm ID to apply to BLAST database as hard masking. Incompatible with: db_soft_mask, subject, subject_loc', equate=False)]
        try:
            self.parameters = extra_parameters + self.parameters
        except AttributeError:
            self.parameters = extra_parameters
        _Ncbiblast2SeqCommandline.__init__(self, cmd, **kwargs)

    def _validate(self):
        incompatibles = {'db_soft_mask': ['db_hard_mask', 'subject', 'subject_loc'], 'db_hard_mask': ['db_soft_mask', 'subject', 'subject_loc']}
        self._validate_incompatibilities(incompatibles)
        _Ncbiblast2SeqCommandline._validate(self)