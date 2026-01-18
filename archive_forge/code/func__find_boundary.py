import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def _find_boundary(self):
    ct_info = tuple((x.strip() for x in self.content_type.split(';')))
    mimetype = ct_info[0]
    if mimetype.split('/')[0].lower() != 'multipart':
        raise NonMultipartContentTypeException("Unexpected mimetype in content-type: '{}'".format(mimetype))
    for item in ct_info[1:]:
        attr, value = _split_on_find(item, '=')
        if attr.lower() == 'boundary':
            self.boundary = encode_with(value.strip('"'), self.encoding)