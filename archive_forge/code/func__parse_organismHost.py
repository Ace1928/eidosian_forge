from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_organismHost(element):
    for organism_element in element:
        if organism_element.tag == NS + 'name':
            append_to_annotations('organism_host', organism_element.text)