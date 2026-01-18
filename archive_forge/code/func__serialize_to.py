from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
def _serialize_to(self, write):
    write('@')
    write(serialize_identifier(self.at_keyword))
    _serialize_to(self.prelude, write)
    if self.content is None:
        write(';')
    else:
        write('{')
        _serialize_to(self.content, write)
        write('}')