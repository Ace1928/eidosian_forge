import fastbencode as bencode
from .. import errors, trace
from .. import transport as _mod_transport
from ..tag import Tags
def _serialize_tag_dict(self, tag_dict):
    td = {k.encode('utf-8'): v for k, v in tag_dict.items()}
    return bencode.bencode(td)