from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class StockholmWriter(SequentialAlignmentWriter):
    """Stockholm/PFAM alignment writer."""
    pfam_gr_mapping = {'secondary_structure': 'SS', 'surface_accessibility': 'SA', 'transmembrane': 'TM', 'posterior_probability': 'PP', 'ligand_binding': 'LI', 'active_site': 'AS', 'intron': 'IN'}
    pfam_gc_mapping = {'reference_annotation': 'RF', 'model_mask': 'MM'}
    pfam_gs_mapping = {'organism': 'OS', 'organism_classification': 'OC', 'look': 'LO'}

    def write_alignment(self, alignment):
        """Use this to write (another) single alignment to an open file.

        Note that sequences and their annotation are recorded
        together (rather than having a block of annotation followed
        by a block of aligned sequences).
        """
        count = len(alignment)
        self._length_of_sequences = alignment.get_alignment_length()
        self._ids_written = []
        if count == 0:
            raise ValueError('Must have at least one sequence')
        if self._length_of_sequences == 0:
            raise ValueError('Non-empty sequences are required')
        self.handle.write('# STOCKHOLM 1.0\n')
        self.handle.write('#=GF SQ %i\n' % count)
        for record in alignment:
            self._write_record(record)
        if alignment.column_annotations:
            for k, v in sorted(alignment.column_annotations.items()):
                if k in self.pfam_gc_mapping:
                    self.handle.write(f'#=GC {self.pfam_gc_mapping[k]} {v}\n')
                elif k in self.pfam_gr_mapping:
                    self.handle.write(f'#=GC {self.pfam_gr_mapping[k]}_cons {v}\n')
                else:
                    pass
        self.handle.write('//\n')

    def _write_record(self, record):
        """Write a single SeqRecord to the file (PRIVATE)."""
        if self._length_of_sequences != len(record.seq):
            raise ValueError('Sequences must all be the same length')
        seq_name = record.id
        if record.name is not None:
            if 'accession' in record.annotations:
                if record.id == record.annotations['accession']:
                    seq_name = record.name
        seq_name = seq_name.replace(' ', '_')
        if 'start' in record.annotations and 'end' in record.annotations:
            suffix = f'/{record.annotations['start']}-{record.annotations['end']}'
            if seq_name[-len(suffix):] != suffix:
                seq_name = '%s/%s-%s' % (seq_name, record.annotations['start'], record.annotations['end'])
        if seq_name in self._ids_written:
            raise ValueError(f'Duplicate record identifier: {seq_name}')
        self._ids_written.append(seq_name)
        self.handle.write(f'{seq_name} {record.seq}\n')
        if 'accession' in record.annotations:
            self.handle.write(f'#=GS {seq_name} AC {self.clean(record.annotations['accession'])}\n')
        elif record.id:
            self.handle.write(f'#=GS {seq_name} AC {self.clean(record.id)}\n')
        if record.description:
            self.handle.write(f'#=GS {seq_name} DE {self.clean(record.description)}\n')
        for xref in record.dbxrefs:
            self.handle.write(f'#=GS {seq_name} DR {self.clean(xref)}\n')
        for key, value in record.annotations.items():
            if key in self.pfam_gs_mapping:
                data = self.clean(str(value))
                if data:
                    self.handle.write('#=GS %s %s %s\n' % (seq_name, self.clean(self.pfam_gs_mapping[key]), data))
            else:
                pass
        for key, value in record.letter_annotations.items():
            if key in self.pfam_gr_mapping and len(str(value)) == len(record.seq):
                data = self.clean(str(value))
                if data:
                    self.handle.write('#=GR %s %s %s\n' % (seq_name, self.clean(self.pfam_gr_mapping[key]), data))
            else:
                pass