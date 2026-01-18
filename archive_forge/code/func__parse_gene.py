from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_gene(element):
    for genename_element in element:
        if 'type' in genename_element.attrib:
            ann_key = 'gene_%s_%s' % (genename_element.tag.replace(NS, ''), genename_element.attrib['type'])
            if genename_element.attrib['type'] == 'primary':
                self.ParsedSeqRecord.annotations[ann_key] = genename_element.text
            else:
                append_to_annotations(ann_key, genename_element.text)