from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_protein(element):
    """Parse protein names (PRIVATE)."""
    descr_set = False
    for protein_element in element:
        if protein_element.tag in [NS + 'recommendedName', NS + 'submittedName', NS + 'alternativeName']:
            for rec_name in protein_element:
                ann_key = '%s_%s' % (protein_element.tag.replace(NS, ''), rec_name.tag.replace(NS, ''))
                append_to_annotations(ann_key, rec_name.text)
                if rec_name.tag == NS + 'fullName' and (not descr_set):
                    self.ParsedSeqRecord.description = rec_name.text
                    descr_set = True
        elif protein_element.tag == NS + 'component':
            pass
        elif protein_element.tag == NS + 'domain':
            pass