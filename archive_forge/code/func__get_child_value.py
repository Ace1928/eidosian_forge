from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _get_child_value(node, name, default=None, error=None):
    children = node.getElementsByTagName(name)
    if children and children[0].childNodes and (children[0].firstChild.nodeType == node.TEXT_NODE):
        return _decode(children[0].firstChild.data)
    elif error:
        raise ValueError(error)
    else:
        return default