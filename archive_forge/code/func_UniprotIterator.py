from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def UniprotIterator(source, alphabet=None, return_raw_comments=False):
    """Iterate over UniProt XML as SeqRecord objects.

    parses an XML entry at a time from any UniProt XML file
    returns a SeqRecord for each iteration

    This generator can be used in Bio.SeqIO

    Argument source is a file-like object or a path to a file.

    Optional argument alphabet should not be used anymore.

    return_raw_comments = True --> comment fields are returned as complete XML to allow further processing
    skip_parsing_errors = True --> if parsing errors are found, skip to next entry
    """
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    try:
        for event, elem in ElementTree.iterparse(source, events=('start', 'start-ns', 'end')):
            if event == 'start-ns' and (not (elem[1].startswith('http://www.w3.org/') or NS == f'{{{elem[1]}}}')):
                raise ValueError(f"SeqIO format 'uniprot-xml' only parses xml with namespace: {NS} but xml has namespace: {{{elem[1]}}}")
            if event == 'end' and elem.tag == NS + 'entry':
                yield Parser(elem, return_raw_comments=return_raw_comments).parse()
                elem.clear()
    except ElementTree.ParseError as exception:
        if errors.messages[exception.code] == errors.XML_ERROR_NO_ELEMENTS:
            assert exception.position == (1, 0)
            raise ValueError('Empty file.') from None
        else:
            raise