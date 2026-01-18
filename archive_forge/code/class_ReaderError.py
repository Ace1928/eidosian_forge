from __future__ import absolute_import
import codecs
from ruamel.yaml.error import YAMLError, FileMark, StringMark, YAMLStreamError
from ruamel.yaml.compat import text_type, binary_type, PY3, UNICODE_SIZE
from ruamel.yaml.util import RegExp
class ReaderError(YAMLError):

    def __init__(self, name, position, character, encoding, reason):
        self.name = name
        self.character = character
        self.position = position
        self.encoding = encoding
        self.reason = reason

    def __str__(self):
        if isinstance(self.character, binary_type):
            return '\'%s\' codec can\'t decode byte #x%02x: %s\n  in "%s", position %d' % (self.encoding, ord(self.character), self.reason, self.name, self.position)
        else:
            return 'unacceptable character #x%04x: %s\n  in "%s", position %d' % (self.character, self.reason, self.name, self.position)