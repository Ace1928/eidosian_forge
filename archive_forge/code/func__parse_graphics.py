from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
def _parse_graphics(element, entry):
    new_graphics = Graphics(entry)
    for k, v in element.attrib.items():
        new_graphics.__setattr__(k, v)
    entry.add_graphics(new_graphics)