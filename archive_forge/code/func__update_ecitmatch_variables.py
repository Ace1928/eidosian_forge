import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def _update_ecitmatch_variables(keywds):
    variables = {'retmode': 'xml'}
    citation_keys = ('journal_title', 'year', 'volume', 'first_page', 'author_name', 'key')
    if isinstance(keywds['bdata'], str):
        variables.update(keywds)
    else:
        variables['db'] = keywds['db']
        bdata = []
        for citation in keywds['bdata']:
            formatted_citation = '|'.join([citation.get(key, '') for key in citation_keys])
            bdata.append(formatted_citation)
        variables['bdata'] = '\r'.join(bdata)
    return variables