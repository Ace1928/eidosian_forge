import sys
import xml.parsers.expat
def _generate_etree_functions(DefusedXMLParser, _TreeBuilder, _parse, _iterparse):
    """Factory for functions needed by etree, dependent on whether
    cElementTree or ElementTree is used."""

    def parse(source, parser=None, forbid_dtd=False, forbid_entities=True, forbid_external=True):
        if parser is None:
            parser = DefusedXMLParser(target=_TreeBuilder(), forbid_dtd=forbid_dtd, forbid_entities=forbid_entities, forbid_external=forbid_external)
        return _parse(source, parser)

    def iterparse(source, events=None, parser=None, forbid_dtd=False, forbid_entities=True, forbid_external=True):
        if parser is None:
            parser = DefusedXMLParser(target=_TreeBuilder(), forbid_dtd=forbid_dtd, forbid_entities=forbid_entities, forbid_external=forbid_external)
        return _iterparse(source, events, parser)

    def fromstring(text, forbid_dtd=False, forbid_entities=True, forbid_external=True):
        parser = DefusedXMLParser(target=_TreeBuilder(), forbid_dtd=forbid_dtd, forbid_entities=forbid_entities, forbid_external=forbid_external)
        parser.feed(text)
        return parser.close()
    return (parse, iterparse, fromstring)