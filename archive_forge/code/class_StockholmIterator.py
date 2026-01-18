from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class StockholmIterator(AlignmentIterator):
    """Loads a Stockholm file from PFAM into MultipleSeqAlignment objects.

    The file may contain multiple concatenated alignments, which are loaded
    and returned incrementally.

    This parser will detect if the Stockholm file follows the PFAM
    conventions for sequence specific meta-data (lines starting #=GS
    and #=GR) and populates the SeqRecord fields accordingly.

    Any annotation which does not follow the PFAM conventions is currently
    ignored.

    If an accession is provided for an entry in the meta data, IT WILL NOT
    be used as the record.id (it will be recorded in the record's
    annotations).  This is because some files have (sub) sequences from
    different parts of the same accession (differentiated by different
    start-end positions).

    Wrap-around alignments are not supported - each sequences must be on
    a single line.  However, interlaced sequences should work.

    For more information on the file format, please see:
    http://sonnhammer.sbc.su.se/Stockholm.html
    https://en.wikipedia.org/wiki/Stockholm_format
    http://bioperl.org/formats/alignment_formats/Stockholm_multiple_alignment_format.html

    For consistency with BioPerl and EMBOSS we call this the "stockholm"
    format.
    """
    pfam_gr_mapping = {'SS': 'secondary_structure', 'SA': 'surface_accessibility', 'TM': 'transmembrane', 'PP': 'posterior_probability', 'LI': 'ligand_binding', 'AS': 'active_site', 'IN': 'intron'}
    pfam_gc_mapping = {'RF': 'reference_annotation', 'MM': 'model_mask'}
    pfam_gs_mapping = {'OS': 'organism', 'OC': 'organism_classification', 'LO': 'look'}
    _header = None

    def __next__(self):
        """Parse the next alignment from the handle."""
        handle = self.handle
        if self._header is None:
            line = handle.readline()
        else:
            line = self._header
            self._header = None
        if not line:
            raise StopIteration
        if line.strip() != '# STOCKHOLM 1.0':
            raise ValueError('Did not find STOCKHOLM header')
        seqs = {}
        ids = {}
        gs = {}
        gr = {}
        gf = {}
        gc = {}
        passed_end_alignment = False
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if line == '# STOCKHOLM 1.0':
                self._header = line
                break
            elif line == '//':
                passed_end_alignment = True
            elif line == '':
                pass
            elif line[0] != '#':
                assert not passed_end_alignment
                parts = [x.strip() for x in line.split(' ', 1)]
                if len(parts) != 2:
                    raise ValueError('Could not split line into identifier and sequence:\n' + line)
                seq_id, seq = parts
                if seq_id not in ids:
                    ids[seq_id] = True
                seqs.setdefault(seq_id, '')
                seqs[seq_id] += seq.replace('.', '-')
            elif len(line) >= 5:
                if line[:5] == '#=GF ':
                    feature, text = line[5:].strip().split(None, 1)
                    if feature not in gf:
                        gf[feature] = [text]
                    else:
                        gf[feature].append(text)
                elif line[:5] == '#=GC ':
                    feature, text = line[5:].strip().split(None, 2)
                    if feature not in gc:
                        gc[feature] = ''
                    gc[feature] += text.strip()
                elif line[:5] == '#=GS ':
                    try:
                        seq_id, feature, text = line[5:].strip().split(None, 2)
                    except ValueError:
                        seq_id, feature = line[5:].strip().split(None, 1)
                        text = ''
                    if seq_id not in gs:
                        gs[seq_id] = {}
                    if feature not in gs[seq_id]:
                        gs[seq_id][feature] = [text]
                    else:
                        gs[seq_id][feature].append(text)
                elif line[:5] == '#=GR ':
                    seq_id, feature, text = line[5:].strip().split(None, 2)
                    if seq_id not in gr:
                        gr[seq_id] = {}
                    if feature not in gr[seq_id]:
                        gr[seq_id][feature] = ''
                    gr[seq_id][feature] += text.strip()
        assert len(seqs) <= len(ids)
        self.ids = ids.keys()
        self.sequences = seqs
        self.seq_annotation = gs
        self.seq_col_annotation = gr
        if ids and seqs:
            if self.records_per_alignment is not None and self.records_per_alignment != len(ids):
                raise ValueError('Found %i records in this alignment, told to expect %i' % (len(ids), self.records_per_alignment))
            alignment_length = len(list(seqs.values())[0])
            records = []
            for seq_id in ids:
                seq = seqs[seq_id]
                if alignment_length != len(seq):
                    raise ValueError('Sequences have different lengths, or repeated identifier')
                name, start, end = self._identifier_split(seq_id)
                record = SeqRecord(Seq(seq), id=seq_id, name=name, description=seq_id, annotations={'accession': name})
                record.annotations['accession'] = name
                if start is not None:
                    record.annotations['start'] = start
                if end is not None:
                    record.annotations['end'] = end
                self._populate_meta_data(seq_id, record)
                records.append(record)
            for k, v in gc.items():
                if len(v) != alignment_length:
                    raise ValueError('%s length %i, expected %i' % (k, len(v), alignment_length))
            alignment = MultipleSeqAlignment(records)
            for k, v in sorted(gc.items()):
                if k in self.pfam_gc_mapping:
                    alignment.column_annotations[self.pfam_gc_mapping[k]] = v
                elif k.endswith('_cons') and k[:-5] in self.pfam_gr_mapping:
                    alignment.column_annotations[self.pfam_gr_mapping[k[:-5]]] = v
                else:
                    alignment.column_annotations['GC:' + k] = v
            alignment._annotations = gr
            return alignment
        else:
            raise StopIteration

    def _identifier_split(self, identifier):
        """Return (name, start, end) string tuple from an identifier (PRIVATE)."""
        if '/' in identifier:
            name, start_end = identifier.rsplit('/', 1)
            if start_end.count('-') == 1:
                try:
                    start, end = start_end.split('-')
                    return (name, int(start), int(end))
                except ValueError:
                    pass
        return (identifier, None, None)

    def _get_meta_data(self, identifier, meta_dict):
        """Take an identifier and returns dict of all meta-data matching it (PRIVATE).

        For example, given "Q9PN73_CAMJE/149-220" will return all matches to
        this or "Q9PN73_CAMJE" which the identifier without its /start-end
        suffix.

        In the example below, the suffix is required to match the AC, but must
        be removed to match the OS and OC meta-data::

            # STOCKHOLM 1.0
            #=GS Q9PN73_CAMJE/149-220  AC Q9PN73
            ...
            Q9PN73_CAMJE/149-220               NKA...
            ...
            #=GS Q9PN73_CAMJE OS Campylobacter jejuni
            #=GS Q9PN73_CAMJE OC Bacteria

        This function will return an empty dictionary if no data is found.
        """
        name, start, end = self._identifier_split(identifier)
        if name == identifier:
            identifier_keys = [identifier]
        else:
            identifier_keys = [identifier, name]
        answer = {}
        for identifier_key in identifier_keys:
            try:
                for feature_key in meta_dict[identifier_key]:
                    answer[feature_key] = meta_dict[identifier_key][feature_key]
            except KeyError:
                pass
        return answer

    def _populate_meta_data(self, identifier, record):
        """Add meta-date to a SecRecord's annotations dictionary (PRIVATE).

        This function applies the PFAM conventions.
        """
        seq_data = self._get_meta_data(identifier, self.seq_annotation)
        for feature in seq_data:
            if feature == 'AC':
                assert len(seq_data[feature]) == 1
                record.annotations['accession'] = seq_data[feature][0]
            elif feature == 'DE':
                record.description = '\n'.join(seq_data[feature])
            elif feature == 'DR':
                record.dbxrefs = seq_data[feature]
            elif feature in self.pfam_gs_mapping:
                record.annotations[self.pfam_gs_mapping[feature]] = ', '.join(seq_data[feature])
            else:
                record.annotations['GS:' + feature] = ', '.join(seq_data[feature])
        seq_col_data = self._get_meta_data(identifier, self.seq_col_annotation)
        for feature in seq_col_data:
            if feature in self.pfam_gr_mapping:
                record.letter_annotations[self.pfam_gr_mapping[feature]] = seq_col_data[feature]
            else:
                record.letter_annotations['GR:' + feature] = seq_col_data[feature]