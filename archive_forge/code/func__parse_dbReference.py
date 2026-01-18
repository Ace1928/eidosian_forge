from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_dbReference(element):
    self.ParsedSeqRecord.dbxrefs.append(element.attrib['type'] + ':' + element.attrib['id'])
    if 'type' in element.attrib:
        if element.attrib['type'] == 'PDB':
            method = ''
            resolution = ''
            for ref_element in element:
                if ref_element.tag == NS + 'property':
                    dat_type = ref_element.attrib['type']
                    if dat_type == 'method':
                        method = ref_element.attrib['value']
                    if dat_type == 'resolution':
                        resolution = ref_element.attrib['value']
                    if dat_type == 'chains':
                        pairs = ref_element.attrib['value'].split(',')
                        for elem in pairs:
                            pair = elem.strip().split('=')
                            if pair[1] != '-':
                                feature = SeqFeature.SeqFeature()
                                feature.type = element.attrib['type']
                                feature.qualifiers['name'] = element.attrib['id']
                                feature.qualifiers['method'] = method
                                feature.qualifiers['resolution'] = resolution
                                feature.qualifiers['chains'] = pair[0].split('/')
                                start = int(pair[1].split('-')[0]) - 1
                                end = int(pair[1].split('-')[1])
                                feature.location = SeqFeature.SimpleLocation(start, end)
    for ref_element in element:
        if ref_element.tag == NS + 'property':
            pass