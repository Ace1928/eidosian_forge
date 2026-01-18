import collections
from requests import compat
def _dump_response_data(response, prefixes, bytearr):
    prefix = prefixes.response
    raw = response.raw
    version_str = HTTP_VERSIONS.get(raw.version, b'?')
    bytearr.extend(prefix + b'HTTP/' + version_str + b' ' + str(raw.status).encode('ascii') + b' ' + _coerce_to_bytes(response.reason) + b'\r\n')
    headers = raw.headers
    for name in headers.keys():
        for value in headers.getlist(name):
            bytearr.extend(prefix + _format_header(name, value))
    bytearr.extend(prefix + b'\r\n')
    bytearr.extend(response.content)