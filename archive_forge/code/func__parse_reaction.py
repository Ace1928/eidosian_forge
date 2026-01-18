from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
def _parse_reaction(element):
    new_reaction = Reaction()
    for k, v in element.attrib.items():
        new_reaction.__setattr__(k, v)
    for subelement in element:
        if subelement.tag == 'substrate':
            new_reaction.add_substrate(int(subelement.attrib['id']))
        elif subelement.tag == 'product':
            new_reaction.add_product(int(subelement.attrib['id']))
    self.pathway.add_reaction(new_reaction)