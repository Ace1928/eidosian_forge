from Bio.Application import _Option, AbstractCommandline, _Switch
class NcbimakeblastdbCommandline(AbstractCommandline):
    """Wrapper for the NCBI BLAST+ program makeblastdb.

    This is a wrapper for the NCBI BLAST+ makeblastdb application
    to create BLAST databases. By default, this creates a blast database
    with the same name as the input file. The default output location
    is the same directory as the input.

    >>> from Bio.Blast.Applications import NcbimakeblastdbCommandline
    >>> cline = NcbimakeblastdbCommandline(dbtype="prot",
    ...                                    input_file="NC_005816.faa")
    >>> cline
    NcbimakeblastdbCommandline(cmd='makeblastdb', dbtype='prot', input_file='NC_005816.faa')
    >>> print(cline)
    makeblastdb -dbtype prot -in NC_005816.faa

    You would typically run the command line with cline() or via the Python
    subprocess module, as described in the Biopython tutorial.
    """

    def __init__(self, cmd='makeblastdb', **kwargs):
        """Initialize the class."""
        self.parameters = [_Switch(['-h', 'h'], 'Print USAGE and DESCRIPTION; ignore other arguments.'), _Switch(['-help', 'help'], 'Print USAGE, DESCRIPTION and ARGUMENTS description; ignore other arguments.'), _Switch(['-version', 'version'], 'Print version number;  ignore other arguments.'), _Option(['-out', 'out'], 'Output file for alignment.', filename=True, equate=False), _Option(['-blastdb_version', 'blastdb_version'], 'Version of BLAST database to be created. Tip: use BLAST database version 4 on 32 bit CPU. Default = 5', equate=False, checker_function=lambda x: x == 4 or x == 5), _Option(['-dbtype', 'dbtype'], "Molecule type of target db ('nucl' or 'prot').", equate=False, is_required=True, checker_function=lambda x: x == 'nucl' or x == 'prot'), _Option(['-in', 'input_file'], 'Input file/database name.', filename=True, equate=False), _Option(['-input_type', 'input_type'], "Type of the data specified in input_file.\n\nDefault = 'fasta'. Added in BLAST 2.2.26.", filename=False, equate=False, checker_function=self._input_type_checker), _Option(['-title', 'title'], 'Title for BLAST database.', filename=False, equate=False), _Switch(['-parse_seqids', 'parse_seqids'], 'Option to parse seqid for FASTA input if set.\n\nFor all other input types, seqids are parsed automatically'), _Switch(['-hash_index', 'hash_index'], 'Create index of sequence hash values.'), _Option(['-mask_data', 'mask_data'], 'Comma-separated list of input files containing masking data as produced by NCBI masking applications (e.g. dustmasker, segmasker, windowmasker).', filename=True, equate=False), _Option(['-mask_id', 'mask_id'], 'Comma-separated list of strings to uniquely identify the masking algorithm.', filename=False, equate=False), _Option(['-mask_desc', 'mask_desc'], 'Comma-separated list of free form strings to describe the masking algorithm details.', filename=False, equate=False), _Switch(['-gi_mask', 'gi_mask'], 'Create GI indexed masking data.'), _Option(['-gi_mask_name', 'gi_mask_name'], 'Comma-separated list of masking data output files.', filename=False, equate=False), _Option(['-max_file_sz', 'max_file_sz'], "Maximum file size for BLAST database files. Default = '1GB'.", filename=False, equate=False), _Option(['-logfile', 'logfile'], 'File to which the program log should be redirected.', filename=True, equate=False), _Option(['-taxid', 'taxid'], 'Taxonomy ID to assign to all sequences.', filename=False, equate=False, checker_function=lambda x: type(x)(int(x)) == x), _Option(['-taxid_map', 'taxid_map'], 'Text file mapping sequence IDs to taxonomy IDs.\n\nFormat:<SequenceId> <TaxonomyId><newline>', filename=True, equate=False)]
        AbstractCommandline.__init__(self, cmd, **kwargs)

    def _input_type_checker(self, command):
        return command in ('asn1_bin', 'asn1_txt', 'blastdb', 'fasta')

    def _validate(self):
        incompatibles = {'mask_id': ['gi_mask'], 'gi_mask': ['mask_id'], 'taxid': ['taxid_map']}
        for a in incompatibles:
            if self._get_parameter(a):
                for b in incompatibles[a]:
                    if self._get_parameter(b):
                        raise ValueError(f'Options {a} and {b} are incompatible.')
        if self.mask_id and (not self.mask_data):
            raise ValueError('Option mask_id requires mask_data to be set.')
        if self.mask_desc and (not self.mask_id):
            raise ValueError('Option mask_desc requires mask_id to be set.')
        if self.gi_mask and (not self.parse_seqids):
            raise ValueError('Option gi_mask requires parse_seqids to be set.')
        if self.gi_mask_name and (not (self.mask_data and self.gi_mask)):
            raise ValueError('Option gi_mask_name requires mask_data and gi_mask to be set.')
        if self.taxid_map and (not self.parse_seqids):
            raise ValueError('Option taxid_map requires parse_seqids to be set.')
        AbstractCommandline._validate(self)