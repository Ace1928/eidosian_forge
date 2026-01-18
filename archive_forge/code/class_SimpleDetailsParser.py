from io import BytesIO
from testtools import content, content_type
from testtools.compat import _b
from subunit import chunked
class SimpleDetailsParser(DetailsParser):
    """Parser for single-part [] delimited details."""

    def __init__(self, state):
        self._message = _b('')
        self._state = state

    def lineReceived(self, line):
        if line == end_marker:
            self._state.endDetails()
            return
        if line[0:2] == quoted_marker:
            self._message += line[1:]
        else:
            self._message += line

    def get_details(self, style=None):
        result = {}
        if not style:
            result['traceback'] = content.Content(content_type.ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [self._message])
        else:
            if style == 'skip':
                name = 'reason'
            else:
                name = 'message'
            result[name] = content.Content(content_type.ContentType('text', 'plain'), lambda: [self._message])
        return result

    def get_message(self):
        return self._message