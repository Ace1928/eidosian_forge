from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
def _parse_relation(element):
    new_relation = Relation()
    new_relation.entry1 = int(element.attrib['entry1'])
    new_relation.entry2 = int(element.attrib['entry2'])
    new_relation.type = element.attrib['type']
    for subtype in element:
        name, value = (subtype.attrib['name'], subtype.attrib['value'])
        if name in ('compound', 'hidden compound'):
            new_relation.subtypes.append((name, int(value)))
        else:
            new_relation.subtypes.append((name, value))
    self.pathway.add_relation(new_relation)