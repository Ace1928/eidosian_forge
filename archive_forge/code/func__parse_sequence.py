from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_sequence(element):
    for k, v in element.attrib.items():
        if k in ('length', 'mass', 'version'):
            self.ParsedSeqRecord.annotations[f'sequence_{k}'] = int(v)
        else:
            self.ParsedSeqRecord.annotations[f'sequence_{k}'] = v
    self.ParsedSeqRecord.seq = Seq(''.join(element.text.split()))
    self.ParsedSeqRecord.annotations['molecule_type'] = 'protein'