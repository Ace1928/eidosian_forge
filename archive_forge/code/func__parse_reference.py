from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_reference(element):
    reference = SeqFeature.Reference()
    authors = []
    scopes = []
    tissues = []
    journal_name = ''
    pub_type = ''
    pub_date = ''
    for ref_element in element:
        if ref_element.tag == NS + 'citation':
            pub_type = ref_element.attrib['type']
            if pub_type == 'submission':
                pub_type += ' to the ' + ref_element.attrib['db']
            if 'name' in ref_element.attrib:
                journal_name = ref_element.attrib['name']
            pub_date = ref_element.attrib.get('date', '')
            j_volume = ref_element.attrib.get('volume', '')
            j_first = ref_element.attrib.get('first', '')
            j_last = ref_element.attrib.get('last', '')
            for cit_element in ref_element:
                if cit_element.tag == NS + 'title':
                    reference.title = cit_element.text
                elif cit_element.tag == NS + 'authorList':
                    for person_element in cit_element:
                        authors.append(person_element.attrib['name'])
                elif cit_element.tag == NS + 'dbReference':
                    self.ParsedSeqRecord.dbxrefs.append(cit_element.attrib['type'] + ':' + cit_element.attrib['id'])
                    if cit_element.attrib['type'] == 'PubMed':
                        reference.pubmed_id = cit_element.attrib['id']
                    elif ref_element.attrib['type'] == 'MEDLINE':
                        reference.medline_id = cit_element.attrib['id']
        elif ref_element.tag == NS + 'scope':
            scopes.append(ref_element.text)
        elif ref_element.tag == NS + 'source':
            for source_element in ref_element:
                if source_element.tag == NS + 'tissue':
                    tissues.append(source_element.text)
    if scopes:
        scopes_str = 'Scope: ' + ', '.join(scopes)
    else:
        scopes_str = ''
    if tissues:
        tissues_str = 'Tissue: ' + ', '.join(tissues)
    else:
        tissues_str = ''
    reference.location = []
    reference.authors = ', '.join(authors)
    if journal_name:
        if pub_date and j_volume and j_first and j_last:
            reference.journal = REFERENCE_JOURNAL % {'name': journal_name, 'volume': j_volume, 'first': j_first, 'last': j_last, 'pub_date': pub_date}
        else:
            reference.journal = journal_name
    reference.comment = ' | '.join((pub_type, pub_date, scopes_str, tissues_str))
    append_to_annotations('references', reference)