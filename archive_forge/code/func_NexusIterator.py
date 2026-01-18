from typing import IO, Iterator, Optional
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentWriter
from Bio.Nexus import Nexus
from Bio.SeqRecord import SeqRecord
def NexusIterator(handle: IO[str], seq_count: Optional[int]=None) -> Iterator[MultipleSeqAlignment]:
    """Return SeqRecord objects from a Nexus file.

    Thus uses the Bio.Nexus module to do the hard work.

    You are expected to call this function via Bio.SeqIO or Bio.AlignIO
    (and not use it directly).

    NOTE - We only expect ONE alignment matrix per Nexus file,
    meaning this iterator will only yield one MultipleSeqAlignment.
    """
    n = Nexus.Nexus(handle)
    if not n.matrix:
        return
    assert len(n.unaltered_taxlabels) == len(n.taxlabels)
    if seq_count and seq_count != len(n.unaltered_taxlabels):
        raise ValueError('Found %i sequences, but seq_count=%i' % (len(n.unaltered_taxlabels), seq_count))
    annotations: Optional[SeqRecord._AnnotationsDict]
    if n.datatype in ('dna', 'nucleotide'):
        annotations = {'molecule_type': 'DNA'}
    elif n.datatype == 'rna':
        annotations = {'molecule_type': 'RNA'}
    elif n.datatype == 'protein':
        annotations = {'molecule_type': 'protein'}
    else:
        annotations = None
    records = (SeqRecord(n.matrix[new_name], id=new_name, name=old_name, description='', annotations=annotations) for old_name, new_name in zip(n.unaltered_taxlabels, n.taxlabels))
    yield MultipleSeqAlignment(records)