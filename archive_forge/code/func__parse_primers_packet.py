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
def _parse_primers_packet(length, data, record):
    """Parse a Primers packet.

    A Primers packet is similar to a Features packet but specifically
    stores primer binding features. The data is a XML string starting
    with a 'Primers' root node.
    """
    xml = parseString(data.decode('UTF-8'))
    for primer in xml.getElementsByTagName('Primer'):
        quals = {}
        name = _get_attribute_value(primer, 'name')
        if name:
            quals['label'] = [name]
        locations = []
        for site in primer.getElementsByTagName('BindingSite'):
            rng = _get_attribute_value(site, 'location', error='Missing binding site location')
            strand = int(_get_attribute_value(site, 'boundStrand', default='0'))
            if strand == 1:
                strand = -1
            else:
                strand = +1
            location = _parse_location(rng, strand, record, is_primer=True)
            simplified = int(_get_attribute_value(site, 'simplified', default='0')) == 1
            if simplified and location in locations:
                continue
            locations.append(location)
            feature = SeqFeature(location, type='primer_bind', qualifiers=quals)
            record.features.append(feature)