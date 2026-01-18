import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
class T1Font(object):
    """Type 1 font class.

    Uses a minimal interpeter that supports just about enough PS to parse
    Type 1 fonts.
    """

    def __init__(self, path, encoding='ascii', kind=None):
        if kind is None:
            self.data, _ = read(path)
        elif kind == 'LWFN':
            self.data = readLWFN(path)
        elif kind == 'PFB':
            self.data = readPFB(path)
        elif kind == 'OTHER':
            self.data = readOther(path)
        else:
            raise ValueError(kind)
        self.encoding = encoding

    def saveAs(self, path, type, dohex=False):
        write(path, self.getData(), type, dohex)

    def getData(self):
        if not hasattr(self, 'data'):
            self.data = self.createData()
        return self.data

    def getGlyphSet(self):
        """Return a generic GlyphSet, which is a dict-like object
        mapping glyph names to glyph objects. The returned glyph objects
        have a .draw() method that supports the Pen protocol, and will
        have an attribute named 'width', but only *after* the .draw() method
        has been called.

        In the case of Type 1, the GlyphSet is simply the CharStrings dict.
        """
        return self['CharStrings']

    def __getitem__(self, key):
        if not hasattr(self, 'font'):
            self.parse()
        return self.font[key]

    def parse(self):
        from fontTools.misc import psLib
        from fontTools.misc import psCharStrings
        self.font = psLib.suckfont(self.data, self.encoding)
        charStrings = self.font['CharStrings']
        lenIV = self.font['Private'].get('lenIV', 4)
        assert lenIV >= 0
        subrs = self.font['Private']['Subrs']
        for glyphName, charString in charStrings.items():
            charString, R = eexec.decrypt(charString, 4330)
            charStrings[glyphName] = psCharStrings.T1CharString(charString[lenIV:], subrs=subrs)
        for i in range(len(subrs)):
            charString, R = eexec.decrypt(subrs[i], 4330)
            subrs[i] = psCharStrings.T1CharString(charString[lenIV:], subrs=subrs)
        del self.data

    def createData(self):
        sf = self.font
        eexec_began = False
        eexec_dict = {}
        lines = []
        lines.extend([self._tobytes(f'%!FontType1-1.1: {sf['FontName']}'), self._tobytes(f'%t1Font: ({fontTools.version})'), self._tobytes(f'%%BeginResource: font {sf['FontName']}')])
        size = 3
        size += 1
        size += 1 + 1
        for key in font_dictionary_keys:
            size += int(key in sf)
        lines.append(self._tobytes(f'{size} dict dup begin'))
        for key, value in sf.items():
            if eexec_began:
                eexec_dict[key] = value
                continue
            if key == 'FontInfo':
                fi = sf['FontInfo']
                size = 3
                for subkey in FontInfo_dictionary_keys:
                    size += int(subkey in fi)
                lines.append(self._tobytes(f'/FontInfo {size} dict dup begin'))
                for subkey, subvalue in fi.items():
                    lines.extend(self._make_lines(subkey, subvalue))
                lines.append(b'end def')
            elif key in _type1_post_eexec_order:
                eexec_dict[key] = value
                eexec_began = True
            else:
                lines.extend(self._make_lines(key, value))
        lines.append(b'end')
        eexec_portion = self.encode_eexec(eexec_dict)
        lines.append(bytesjoin([b'currentfile eexec ', eexec_portion]))
        for _ in range(8):
            lines.append(self._tobytes('0' * 64))
        lines.extend([b'cleartomark', b'%%EndResource', b'%%EOF'])
        data = bytesjoin(lines, '\n')
        return data

    def encode_eexec(self, eexec_dict):
        lines = []
        RD_key, ND_key, NP_key = (None, None, None)
        lenIV = 4
        subrs = std_subrs
        sortedItems = sorted(eexec_dict.items(), key=lambda item: item[0] != 'Private')
        for key, value in sortedItems:
            if key == 'Private':
                pr = eexec_dict['Private']
                size = 3
                for subkey in Private_dictionary_keys:
                    size += int(subkey in pr)
                lines.append(b'dup /Private')
                lines.append(self._tobytes(f'{size} dict dup begin'))
                for subkey, subvalue in pr.items():
                    if not RD_key and subvalue == RD_value:
                        RD_key = subkey
                    elif not ND_key and subvalue in ND_values:
                        ND_key = subkey
                    elif not NP_key and subvalue in PD_values:
                        NP_key = subkey
                    if subkey == 'lenIV':
                        lenIV = subvalue
                    if subkey == 'OtherSubrs':
                        lines.append(self._tobytes(hintothers))
                    elif subkey == 'Subrs':
                        for subr_bin in subvalue:
                            subr_bin.compile()
                        subrs = [subr_bin.bytecode for subr_bin in subvalue]
                        lines.append(f'/Subrs {len(subrs)} array'.encode('ascii'))
                        for i, subr_bin in enumerate(subrs):
                            encrypted_subr, R = eexec.encrypt(bytesjoin([char_IV[:lenIV], subr_bin]), 4330)
                            lines.append(bytesjoin([self._tobytes(f'dup {i} {len(encrypted_subr)} {RD_key} '), encrypted_subr, self._tobytes(f' {NP_key}')]))
                        lines.append(b'def')
                        lines.append(b'put')
                    else:
                        lines.extend(self._make_lines(subkey, subvalue))
            elif key == 'CharStrings':
                lines.append(b'dup /CharStrings')
                lines.append(self._tobytes(f'{len(eexec_dict['CharStrings'])} dict dup begin'))
                for glyph_name, char_bin in eexec_dict['CharStrings'].items():
                    char_bin.compile()
                    encrypted_char, R = eexec.encrypt(bytesjoin([char_IV[:lenIV], char_bin.bytecode]), 4330)
                    lines.append(bytesjoin([self._tobytes(f'/{glyph_name} {len(encrypted_char)} {RD_key} '), encrypted_char, self._tobytes(f' {ND_key}')]))
                lines.append(b'end put')
            else:
                lines.extend(self._make_lines(key, value))
        lines.extend([b'end', b'dup /FontName get exch definefont pop', b'mark', b'currentfile closefile\n'])
        eexec_portion = bytesjoin(lines, '\n')
        encrypted_eexec, R = eexec.encrypt(bytesjoin([eexec_IV, eexec_portion]), 55665)
        return encrypted_eexec

    def _make_lines(self, key, value):
        if key == 'FontName':
            return [self._tobytes(f'/{key} /{value} def')]
        if key in ['isFixedPitch', 'ForceBold', 'RndStemUp']:
            return [self._tobytes(f'/{key} {('true' if value else 'false')} def')]
        elif key == 'Encoding':
            if value == StandardEncoding:
                return [self._tobytes(f'/{key} StandardEncoding def')]
            else:
                lines = []
                lines.append(b'/Encoding 256 array')
                lines.append(b'0 1 255 {1 index exch /.notdef put} for')
                for i in range(256):
                    name = value[i]
                    if name != '.notdef':
                        lines.append(self._tobytes(f'dup {i} /{name} put'))
                lines.append(b'def')
                return lines
        if isinstance(value, str):
            return [self._tobytes(f'/{key} ({value}) def')]
        elif isinstance(value, bool):
            return [self._tobytes(f'/{key} {('true' if value else 'false')} def')]
        elif isinstance(value, list):
            return [self._tobytes(f'/{key} [{' '.join((str(v) for v in value))}] def')]
        elif isinstance(value, tuple):
            return [self._tobytes(f'/{key} {{{' '.join((str(v) for v in value))}}} def')]
        else:
            return [self._tobytes(f'/{key} {value} def')]

    def _tobytes(self, s, errors='strict'):
        return tobytes(s, self.encoding, errors)