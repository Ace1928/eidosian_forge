from ctypes import *
from .base import FontException
import pyglet.lib
class FreeTypeError(FontException):

    def __init__(self, message, errcode):
        self.message = message
        self.errcode = errcode

    def __str__(self):
        return '%s: %s (%s)' % (self.__class__.__name__, self.message, self._ft_errors.get(self.errcode, 'unknown error'))

    @classmethod
    def check_and_raise_on_error(cls, errcode):
        if errcode != 0:
            raise cls(None, errcode)
    _ft_errors = {0: 'no error', 1: 'cannot open resource', 2: 'unknown file format', 3: 'broken file', 4: 'invalid FreeType version', 5: 'module version is too low', 6: 'invalid argument', 7: 'unimplemented feature', 8: 'broken table', 9: 'broken offset within table', 16: 'invalid glyph index', 17: 'invalid character code', 18: 'unsupported glyph image format', 19: 'cannot render this glyph format', 20: 'invalid outline', 21: 'invalid composite glyph', 22: 'too many hints', 23: 'invalid pixel size', 32: 'invalid object handle', 33: 'invalid library handle', 34: 'invalid module handle', 35: 'invalid face handle', 36: 'invalid size handle', 37: 'invalid glyph slot handle', 38: 'invalid charmap handle', 39: 'invalid cache manager handle', 40: 'invalid stream handle', 48: 'too many modules', 49: 'too many extensions', 64: 'out of memory', 65: 'unlisted object', 81: 'cannot open stream', 82: 'invalid stream seek', 83: 'invalid stream skip', 84: 'invalid stream read', 85: 'invalid stream operation', 86: 'invalid frame operation', 87: 'nested frame access', 88: 'invalid frame read', 96: 'raster uninitialized', 97: 'raster corrupted', 98: 'raster overflow', 99: 'negative height while rastering', 112: 'too many registered caches', 128: 'invalid opcode', 129: 'too few arguments', 130: 'stack overflow', 131: 'code overflow', 132: 'bad argument', 133: 'division by zero', 134: 'invalid reference', 135: 'found debug opcode', 136: 'found ENDF opcode in execution stream', 137: 'nested DEFS', 138: 'invalid code range', 139: 'execution context too long', 140: 'too many function definitions', 141: 'too many instruction definitions', 142: 'SFNT font table missing', 143: 'horizontal header (hhea, table missing', 144: 'locations (loca, table missing', 145: 'name table missing', 146: 'character map (cmap, table missing', 147: 'horizontal metrics (hmtx, table missing', 148: 'PostScript (post, table missing', 149: 'invalid horizontal metrics', 150: 'invalid character map (cmap, format', 151: 'invalid ppem value', 152: 'invalid vertical metrics', 153: 'could not find context', 154: 'invalid PostScript (post, table format', 155: 'invalid PostScript (post, table', 160: 'opcode syntax error', 161: 'argument stack underflow', 162: 'ignore', 176: "`STARTFONT' field missing", 177: "`FONT' field missing", 178: "`SIZE' field missing", 179: "`CHARS' field missing", 180: "`STARTCHAR' field missing", 181: "`ENCODING' field missing", 182: "`BBX' field missing", 183: "`BBX' too big"}