import warnings
from io import StringIO
import time
from urllib.parse import urlencode
from urllib.request import build_opener, install_opener
from urllib.request import urlopen
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler
from urllib.request import Request
from Bio import BiopythonWarning
from Bio._utils import function_with_previous
def _parse_qblast_ref_page(handle):
    """Extract a tuple of RID, RTOE from the 'please wait' page (PRIVATE).

    The NCBI FAQ pages use TOE for 'Time of Execution', so RTOE is probably
    'Request Time of Execution' and RID would be 'Request Identifier'.
    """
    s = handle.read().decode()
    i = s.find('RID =')
    if i == -1:
        rid = None
    else:
        j = s.find('\n', i)
        rid = s[i + len('RID ='):j].strip()
    i = s.find('RTOE =')
    if i == -1:
        rtoe = None
    else:
        j = s.find('\n', i)
        rtoe = s[i + len('RTOE ='):j].strip()
    if not rid and (not rtoe):
        i = s.find('<div class="error msInf">')
        if i != -1:
            msg = s[i + len('<div class="error msInf">'):].strip()
            msg = msg.split('</div>', 1)[0].split('\n', 1)[0].strip()
            if msg:
                raise ValueError(f'Error message from NCBI: {msg}')
        i = s.find('<p class="error">')
        if i != -1:
            msg = s[i + len('<p class="error">'):].strip()
            msg = msg.split('</p>', 1)[0].split('\n', 1)[0].strip()
            if msg:
                raise ValueError(f'Error message from NCBI: {msg}')
        i = s.find('Message ID#')
        if i != -1:
            msg = s[i:].split('<', 1)[0].split('\n', 1)[0].strip()
            raise ValueError(f'Error message from NCBI: {msg}')
        raise ValueError("No RID and no RTOE found in the 'please wait' page, there was probably an error in your request but we could not extract a helpful error message.")
    elif not rid:
        raise ValueError(f"No RID found in the 'please wait' page. (although RTOE = {rtoe!r})")
    elif not rtoe:
        raise ValueError(f"No RTOE found in the 'please wait' page. (although RID = {rid!r})")
    try:
        return (rid, int(rtoe))
    except ValueError:
        raise ValueError(f"A non-integer RTOE found in the 'please wait' page, {rtoe!r}") from None