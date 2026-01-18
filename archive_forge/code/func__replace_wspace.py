from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def _replace_wspace(text):
    """Replace tab, LF and CR characters with spaces, but don't collapse (PRIVATE).

    See "Replace Whitespace Policy" in the phyloXML spec glossary:
    http://phyloxml.org/documentation/version_100/phyloxml.xsd.html#Glossary
    """
    for char in ('\t', '\n', '\r'):
        if char in text:
            text = text.replace(char, ' ')
    return text