import xml.etree.ElementTree as ET
from Bio.motifs import meme
def __read_sequences(record, xml_tree):
    """Read sequences from XML ElementTree object."""
    for sequence_tree in xml_tree.find('sequences').findall('sequence'):
        sequence_name = sequence_tree.get('name')
        record.sequences.append(sequence_name)
        diagram_str = __make_diagram(record, sequence_tree)
        record.diagrams[sequence_name] = diagram_str