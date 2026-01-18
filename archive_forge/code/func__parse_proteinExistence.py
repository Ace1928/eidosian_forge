from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_proteinExistence(element):
    append_to_annotations('proteinExistence', element.attrib['type'])