from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_geneLocation(element):
    append_to_annotations('geneLocation', element.attrib['type'])