from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
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