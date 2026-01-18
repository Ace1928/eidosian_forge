import io
from .. import ui
from ..ui import text as ui_text
class StringIOWithEncoding(io.StringIO):
    encoding = 'ascii'

    def write(self, string):
        if isinstance(string, bytes):
            string = string.decode(self.encoding)
        io.StringIO.write(self, string)