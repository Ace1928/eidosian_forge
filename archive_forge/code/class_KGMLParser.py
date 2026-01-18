from xml.etree import ElementTree
from io import StringIO
from Bio.KEGG.KGML.KGML_pathway import Component, Entry, Graphics
from Bio.KEGG.KGML.KGML_pathway import Pathway, Reaction, Relation
class KGMLParser:
    """Parses a KGML XML Pathway entry into a Pathway object.

    Example: Read and parse large metabolism file

    >>> from Bio.KEGG.KGML.KGML_parser import read
    >>> pathway = read(open('KEGG/ko01100.xml', 'r'))
    >>> print(len(pathway.entries))
    3628
    >>> print(len(pathway.reactions))
    1672
    >>> print(len(pathway.maps))
    149

    >>> pathway = read(open('KEGG/ko00010.xml', 'r'))
    >>> print(pathway) #doctest: +NORMALIZE_WHITESPACE
    Pathway: Glycolysis / Gluconeogenesis
    KEGG ID: path:ko00010
    Image file: http://www.kegg.jp/kegg/pathway/ko/ko00010.png
    Organism: ko
    Entries: 99
    Entry types:
        ortholog: 61
        compound: 31
        map: 7

    """

    def __init__(self, elem):
        """Initialize the class."""
        self.entry = elem

    def parse(self):
        """Parse the input elements."""

        def _parse_pathway(attrib):
            for k, v in attrib.items():
                self.pathway.__setattr__(k, v)

        def _parse_entry(element):
            new_entry = Entry()
            for k, v in element.attrib.items():
                new_entry.__setattr__(k, v)
            for subelement in element:
                if subelement.tag == 'graphics':
                    _parse_graphics(subelement, new_entry)
                elif subelement.tag == 'component':
                    _parse_component(subelement, new_entry)
            self.pathway.add_entry(new_entry)

        def _parse_graphics(element, entry):
            new_graphics = Graphics(entry)
            for k, v in element.attrib.items():
                new_graphics.__setattr__(k, v)
            entry.add_graphics(new_graphics)

        def _parse_component(element, entry):
            new_component = Component(entry)
            for k, v in element.attrib.items():
                new_component.__setattr__(k, v)
            entry.add_component(new_component)

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
        self.pathway = Pathway()
        _parse_pathway(self.entry.attrib)
        for element in self.entry:
            if element.tag == 'entry':
                _parse_entry(element)
            elif element.tag == 'reaction':
                _parse_reaction(element)
            elif element.tag == 'relation':
                _parse_relation(element)
            else:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn(f'Warning: tag {element.tag} not implemented in parser', BiopythonParserWarning)
        return self.pathway