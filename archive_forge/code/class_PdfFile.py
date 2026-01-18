import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
class PdfFile:
    """PDF file object."""

    def __init__(self, filename, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict from strings to strings and dates
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.
        """
        super().__init__()
        self._object_seq = itertools.count(1)
        self.xrefTable = [[0, 65535, 'the zero object']]
        self.passed_in_file_object = False
        self.original_file_like = None
        self.tell_base = 0
        fh, opened = cbook.to_filehandle(filename, 'wb', return_opened=True)
        if not opened:
            try:
                self.tell_base = filename.tell()
            except OSError:
                fh = BytesIO()
                self.original_file_like = filename
            else:
                fh = filename
                self.passed_in_file_object = True
        self.fh = fh
        self.currentstream = None
        fh.write(b'%PDF-1.4\n')
        fh.write(b'%\xac\xdc \xab\xba\n')
        self.rootObject = self.reserveObject('root')
        self.pagesObject = self.reserveObject('pages')
        self.pageList = []
        self.fontObject = self.reserveObject('fonts')
        self._extGStateObject = self.reserveObject('extended graphics states')
        self.hatchObject = self.reserveObject('tiling patterns')
        self.gouraudObject = self.reserveObject('Gouraud triangles')
        self.XObjectObject = self.reserveObject('external objects')
        self.resourceObject = self.reserveObject('resources')
        root = {'Type': Name('Catalog'), 'Pages': self.pagesObject}
        self.writeObject(self.rootObject, root)
        self.infoDict = _create_pdf_info_dict('pdf', metadata or {})
        self.fontNames = {}
        self._internal_font_seq = (Name(f'F{i}') for i in itertools.count(1))
        self.dviFontInfo = {}
        self.type1Descriptors = {}
        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self.alphaStates = {}
        self._alpha_state_seq = (Name(f'A{i}') for i in itertools.count(1))
        self._soft_mask_states = {}
        self._soft_mask_seq = (Name(f'SM{i}') for i in itertools.count(1))
        self._soft_mask_groups = []
        self.hatchPatterns = {}
        self._hatch_pattern_seq = (Name(f'H{i}') for i in itertools.count(1))
        self.gouraudTriangles = []
        self._images = {}
        self._image_seq = (Name(f'I{i}') for i in itertools.count(1))
        self.markers = {}
        self.multi_byte_charprocs = {}
        self.paths = []
        self._annotations = []
        self.pageAnnotations = []
        procsets = [Name(x) for x in 'PDF Text ImageB ImageC ImageI'.split()]
        resources = {'Font': self.fontObject, 'XObject': self.XObjectObject, 'ExtGState': self._extGStateObject, 'Pattern': self.hatchObject, 'Shading': self.gouraudObject, 'ProcSet': procsets}
        self.writeObject(self.resourceObject, resources)

    def newPage(self, width, height):
        self.endStream()
        self.width, self.height = (width, height)
        contentObject = self.reserveObject('page contents')
        annotsObject = self.reserveObject('annotations')
        thePage = {'Type': Name('Page'), 'Parent': self.pagesObject, 'Resources': self.resourceObject, 'MediaBox': [0, 0, 72 * width, 72 * height], 'Contents': contentObject, 'Annots': annotsObject}
        pageObject = self.reserveObject('page')
        self.writeObject(pageObject, thePage)
        self.pageList.append(pageObject)
        self._annotations.append((annotsObject, self.pageAnnotations))
        self.beginStream(contentObject.id, self.reserveObject('length of content stream'))
        self.output(Name('DeviceRGB'), Op.setcolorspace_stroke)
        self.output(Name('DeviceRGB'), Op.setcolorspace_nonstroke)
        self.output(GraphicsContextPdf.joinstyles['round'], Op.setlinejoin)
        self.pageAnnotations = []

    def newTextnote(self, text, positionRect=[-100, -100, 0, 0]):
        theNote = {'Type': Name('Annot'), 'Subtype': Name('Text'), 'Contents': text, 'Rect': positionRect}
        self.pageAnnotations.append(theNote)

    def _get_subsetted_psname(self, ps_name, charmap):

        def toStr(n, base):
            if n < base:
                return string.ascii_uppercase[n]
            else:
                return toStr(n // base, base) + string.ascii_uppercase[n % base]
        hashed = hash(frozenset(charmap.keys())) % ((sys.maxsize + 1) * 2)
        prefix = toStr(hashed, 26)
        return prefix[:6] + '+' + ps_name

    def finalize(self):
        """Write out the various deferred objects and the pdf end matter."""
        self.endStream()
        self._write_annotations()
        self.writeFonts()
        self.writeExtGSTates()
        self._write_soft_mask_groups()
        self.writeHatches()
        self.writeGouraudTriangles()
        xobjects = {name: ob for image, name, ob in self._images.values()}
        for tup in self.markers.values():
            xobjects[tup[0]] = tup[1]
        for name, value in self.multi_byte_charprocs.items():
            xobjects[name] = value
        for name, path, trans, ob, join, cap, padding, filled, stroked in self.paths:
            xobjects[name] = ob
        self.writeObject(self.XObjectObject, xobjects)
        self.writeImages()
        self.writeMarkers()
        self.writePathCollectionTemplates()
        self.writeObject(self.pagesObject, {'Type': Name('Pages'), 'Kids': self.pageList, 'Count': len(self.pageList)})
        self.writeInfoDict()
        self.writeXref()
        self.writeTrailer()

    def close(self):
        """Flush all buffers and free all resources."""
        self.endStream()
        if self.passed_in_file_object:
            self.fh.flush()
        else:
            if self.original_file_like is not None:
                self.original_file_like.write(self.fh.getvalue())
            self.fh.close()

    def write(self, data):
        if self.currentstream is None:
            self.fh.write(data)
        else:
            self.currentstream.write(data)

    def output(self, *data):
        self.write(_fill([pdfRepr(x) for x in data]))
        self.write(b'\n')

    def beginStream(self, id, len, extra=None, png=None):
        assert self.currentstream is None
        self.currentstream = Stream(id, len, self, extra, png)

    def endStream(self):
        if self.currentstream is not None:
            self.currentstream.end()
            self.currentstream = None

    def outputStream(self, ref, data, *, extra=None):
        self.beginStream(ref.id, None, extra)
        self.currentstream.write(data)
        self.endStream()

    def _write_annotations(self):
        for annotsObject, annotations in self._annotations:
            self.writeObject(annotsObject, annotations)

    def fontName(self, fontprop):
        """
        Select a font based on fontprop and return a name suitable for
        Op.selectfont. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """
        if isinstance(fontprop, str):
            filenames = [fontprop]
        elif mpl.rcParams['pdf.use14corefonts']:
            filenames = _fontManager._find_fonts_by_props(fontprop, fontext='afm', directory=RendererPdf._afm_font_dir)
        else:
            filenames = _fontManager._find_fonts_by_props(fontprop)
        first_Fx = None
        for fname in filenames:
            Fx = self.fontNames.get(fname)
            if not first_Fx:
                first_Fx = Fx
            if Fx is None:
                Fx = next(self._internal_font_seq)
                self.fontNames[fname] = Fx
                _log.debug('Assigning font %s = %r', Fx, fname)
                if not first_Fx:
                    first_Fx = Fx
        return first_Fx

    def dviFontName(self, dvifont):
        """
        Given a dvi font object, return a name suitable for Op.selectfont.
        This registers the font information in ``self.dviFontInfo`` if not yet
        registered.
        """
        dvi_info = self.dviFontInfo.get(dvifont.texname)
        if dvi_info is not None:
            return dvi_info.pdfname
        tex_font_map = dviread.PsfontsMap(dviread.find_tex_file('pdftex.map'))
        psfont = tex_font_map[dvifont.texname]
        if psfont.filename is None:
            raise ValueError('No usable font file found for {} (TeX: {}); the font may lack a Type-1 version'.format(psfont.psname, dvifont.texname))
        pdfname = next(self._internal_font_seq)
        _log.debug('Assigning font %s = %s (dvi)', pdfname, dvifont.texname)
        self.dviFontInfo[dvifont.texname] = types.SimpleNamespace(dvifont=dvifont, pdfname=pdfname, fontfile=psfont.filename, basefont=psfont.psname, encodingfile=psfont.encoding, effects=psfont.effects)
        return pdfname

    def writeFonts(self):
        fonts = {}
        for dviname, info in sorted(self.dviFontInfo.items()):
            Fx = info.pdfname
            _log.debug('Embedding Type-1 font %s from dvi.', dviname)
            fonts[Fx] = self._embedTeXFont(info)
        for filename in sorted(self.fontNames):
            Fx = self.fontNames[filename]
            _log.debug('Embedding font %s.', filename)
            if filename.endswith('.afm'):
                _log.debug('Writing AFM font.')
                fonts[Fx] = self._write_afm_font(filename)
            else:
                _log.debug('Writing TrueType font.')
                chars = self._character_tracker.used.get(filename)
                if chars:
                    fonts[Fx] = self.embedTTF(filename, chars)
        self.writeObject(self.fontObject, fonts)

    def _write_afm_font(self, filename):
        with open(filename, 'rb') as fh:
            font = AFM(fh)
        fontname = font.get_fontname()
        fontdict = {'Type': Name('Font'), 'Subtype': Name('Type1'), 'BaseFont': Name(fontname), 'Encoding': Name('WinAnsiEncoding')}
        fontdictObject = self.reserveObject('font dictionary')
        self.writeObject(fontdictObject, fontdict)
        return fontdictObject

    def _embedTeXFont(self, fontinfo):
        _log.debug('Embedding TeX font %s - fontinfo=%s', fontinfo.dvifont.texname, fontinfo.__dict__)
        widthsObject = self.reserveObject('font widths')
        self.writeObject(widthsObject, fontinfo.dvifont.widths)
        fontdictObject = self.reserveObject('font dictionary')
        fontdict = {'Type': Name('Font'), 'Subtype': Name('Type1'), 'FirstChar': 0, 'LastChar': len(fontinfo.dvifont.widths) - 1, 'Widths': widthsObject}
        if fontinfo.encodingfile is not None:
            fontdict['Encoding'] = {'Type': Name('Encoding'), 'Differences': [0, *map(Name, dviread._parse_enc(fontinfo.encodingfile))]}
        if fontinfo.fontfile is None:
            _log.warning('Because of TeX configuration (pdftex.map, see updmap option pdftexDownloadBase14) the font %s is not embedded. This is deprecated as of PDF 1.5 and it may cause the consumer application to show something that was not intended.', fontinfo.basefont)
            fontdict['BaseFont'] = Name(fontinfo.basefont)
            self.writeObject(fontdictObject, fontdict)
            return fontdictObject
        t1font = _type1font.Type1Font(fontinfo.fontfile)
        if fontinfo.effects:
            t1font = t1font.transform(fontinfo.effects)
        fontdict['BaseFont'] = Name(t1font.prop['FontName'])
        effects = (fontinfo.effects.get('slant', 0.0), fontinfo.effects.get('extend', 1.0))
        fontdesc = self.type1Descriptors.get((fontinfo.fontfile, effects))
        if fontdesc is None:
            fontdesc = self.createType1Descriptor(t1font, fontinfo.fontfile)
            self.type1Descriptors[fontinfo.fontfile, effects] = fontdesc
        fontdict['FontDescriptor'] = fontdesc
        self.writeObject(fontdictObject, fontdict)
        return fontdictObject

    def createType1Descriptor(self, t1font, fontfile):
        fontdescObject = self.reserveObject('font descriptor')
        fontfileObject = self.reserveObject('font file')
        italic_angle = t1font.prop['ItalicAngle']
        fixed_pitch = t1font.prop['isFixedPitch']
        flags = 0
        if fixed_pitch:
            flags |= 1 << 0
        if 0:
            flags |= 1 << 1
        if 1:
            flags |= 1 << 2
        else:
            flags |= 1 << 5
        if italic_angle:
            flags |= 1 << 6
        if 0:
            flags |= 1 << 16
        if 0:
            flags |= 1 << 17
        if 0:
            flags |= 1 << 18
        ft2font = get_font(fontfile)
        descriptor = {'Type': Name('FontDescriptor'), 'FontName': Name(t1font.prop['FontName']), 'Flags': flags, 'FontBBox': ft2font.bbox, 'ItalicAngle': italic_angle, 'Ascent': ft2font.ascender, 'Descent': ft2font.descender, 'CapHeight': 1000, 'XHeight': 500, 'FontFile': fontfileObject, 'FontFamily': t1font.prop['FamilyName'], 'StemV': 50}
        self.writeObject(fontdescObject, descriptor)
        self.outputStream(fontfileObject, b''.join(t1font.parts[:2]), extra={'Length1': len(t1font.parts[0]), 'Length2': len(t1font.parts[1]), 'Length3': 0})
        return fontdescObject

    def _get_xobject_glyph_name(self, filename, glyph_name):
        Fx = self.fontName(filename)
        return '-'.join([Fx.name.decode(), os.path.splitext(os.path.basename(filename))[0], glyph_name])
    _identityToUnicodeCMap = b'/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo\n<< /Registry (Adobe)\n   /Ordering (UCS)\n   /Supplement 0\n>> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n%d beginbfrange\n%s\nendbfrange\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend'

    def embedTTF(self, filename, characters):
        """Embed the TTF font from the named file into the document."""
        font = get_font(filename)
        fonttype = mpl.rcParams['pdf.fonttype']

        def cvt(length, upe=font.units_per_EM, nearest=True):
            """Convert font coordinates to PDF glyph coordinates."""
            value = length / upe * 1000
            if nearest:
                return round(value)
            if value < 0:
                return math.floor(value)
            else:
                return math.ceil(value)

        def embedTTFType3(font, characters, descriptor):
            """The Type 3-specific part of embedding a Truetype font"""
            widthsObject = self.reserveObject('font widths')
            fontdescObject = self.reserveObject('font descriptor')
            fontdictObject = self.reserveObject('font dictionary')
            charprocsObject = self.reserveObject('character procs')
            differencesArray = []
            firstchar, lastchar = (0, 255)
            bbox = [cvt(x, nearest=False) for x in font.bbox]
            fontdict = {'Type': Name('Font'), 'BaseFont': ps_name, 'FirstChar': firstchar, 'LastChar': lastchar, 'FontDescriptor': fontdescObject, 'Subtype': Name('Type3'), 'Name': descriptor['FontName'], 'FontBBox': bbox, 'FontMatrix': [0.001, 0, 0, 0.001, 0, 0], 'CharProcs': charprocsObject, 'Encoding': {'Type': Name('Encoding'), 'Differences': differencesArray}, 'Widths': widthsObject}
            from encodings import cp1252

            def get_char_width(charcode):
                s = ord(cp1252.decoding_table[charcode])
                width = font.load_char(s, flags=LOAD_NO_SCALE | LOAD_NO_HINTING).horiAdvance
                return cvt(width)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                widths = [get_char_width(charcode) for charcode in range(firstchar, lastchar + 1)]
            descriptor['MaxWidth'] = max(widths)
            glyph_ids = []
            differences = []
            multi_byte_chars = set()
            for c in characters:
                ccode = c
                gind = font.get_char_index(ccode)
                glyph_ids.append(gind)
                glyph_name = font.get_glyph_name(gind)
                if ccode <= 255:
                    differences.append((ccode, glyph_name))
                else:
                    multi_byte_chars.add(glyph_name)
            differences.sort()
            last_c = -2
            for c, name in differences:
                if c != last_c + 1:
                    differencesArray.append(c)
                differencesArray.append(Name(name))
                last_c = c
            rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
            charprocs = {}
            for charname in sorted(rawcharprocs):
                stream = rawcharprocs[charname]
                charprocDict = {}
                if charname in multi_byte_chars:
                    charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
                    stream = stream[stream.find(b'd1') + 2:]
                charprocObject = self.reserveObject('charProc')
                self.outputStream(charprocObject, stream, extra=charprocDict)
                if charname in multi_byte_chars:
                    name = self._get_xobject_glyph_name(filename, charname)
                    self.multi_byte_charprocs[name] = charprocObject
                else:
                    charprocs[charname] = charprocObject
            self.writeObject(fontdictObject, fontdict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(widthsObject, widths)
            self.writeObject(charprocsObject, charprocs)
            return fontdictObject

        def embedTTFType42(font, characters, descriptor):
            """The Type 42-specific part of embedding a Truetype font"""
            fontdescObject = self.reserveObject('font descriptor')
            cidFontDictObject = self.reserveObject('CID font dictionary')
            type0FontDictObject = self.reserveObject('Type 0 font dictionary')
            cidToGidMapObject = self.reserveObject('CIDToGIDMap stream')
            fontfileObject = self.reserveObject('font file stream')
            wObject = self.reserveObject('Type 0 widths')
            toUnicodeMapObject = self.reserveObject('ToUnicode map')
            subset_str = ''.join((chr(c) for c in characters))
            _log.debug('SUBSET %s characters: %s', filename, subset_str)
            fontdata = _backend_pdf_ps.get_glyphs_subset(filename, subset_str)
            _log.debug('SUBSET %s %d -> %d', filename, os.stat(filename).st_size, fontdata.getbuffer().nbytes)
            full_font = font
            font = FT2Font(fontdata)
            cidFontDict = {'Type': Name('Font'), 'Subtype': Name('CIDFontType2'), 'BaseFont': ps_name, 'CIDSystemInfo': {'Registry': 'Adobe', 'Ordering': 'Identity', 'Supplement': 0}, 'FontDescriptor': fontdescObject, 'W': wObject, 'CIDToGIDMap': cidToGidMapObject}
            type0FontDict = {'Type': Name('Font'), 'Subtype': Name('Type0'), 'BaseFont': ps_name, 'Encoding': Name('Identity-H'), 'DescendantFonts': [cidFontDictObject], 'ToUnicode': toUnicodeMapObject}
            descriptor['FontFile2'] = fontfileObject
            self.outputStream(fontfileObject, fontdata.getvalue(), extra={'Length1': fontdata.getbuffer().nbytes})
            cid_to_gid_map = ['\x00'] * 65536
            widths = []
            max_ccode = 0
            for c in characters:
                ccode = c
                gind = font.get_char_index(ccode)
                glyph = font.load_char(ccode, flags=LOAD_NO_SCALE | LOAD_NO_HINTING)
                widths.append((ccode, cvt(glyph.horiAdvance)))
                if ccode < 65536:
                    cid_to_gid_map[ccode] = chr(gind)
                max_ccode = max(ccode, max_ccode)
            widths.sort()
            cid_to_gid_map = cid_to_gid_map[:max_ccode + 1]
            last_ccode = -2
            w = []
            max_width = 0
            unicode_groups = []
            for ccode, width in widths:
                if ccode != last_ccode + 1:
                    w.append(ccode)
                    w.append([width])
                    unicode_groups.append([ccode, ccode])
                else:
                    w[-1].append(width)
                    unicode_groups[-1][1] = ccode
                max_width = max(max_width, width)
                last_ccode = ccode
            unicode_bfrange = []
            for start, end in unicode_groups:
                if start > 65535:
                    continue
                end = min(65535, end)
                unicode_bfrange.append(b'<%04x> <%04x> [%s]' % (start, end, b' '.join((b'<%04x>' % x for x in range(start, end + 1)))))
            unicode_cmap = self._identityToUnicodeCMap % (len(unicode_groups), b'\n'.join(unicode_bfrange))
            glyph_ids = []
            for ccode in characters:
                if not _font_supports_glyph(fonttype, ccode):
                    gind = full_font.get_char_index(ccode)
                    glyph_ids.append(gind)
            bbox = [cvt(x, nearest=False) for x in full_font.bbox]
            rawcharprocs = _get_pdf_charprocs(filename, glyph_ids)
            for charname in sorted(rawcharprocs):
                stream = rawcharprocs[charname]
                charprocDict = {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': bbox}
                stream = stream[stream.find(b'd1') + 2:]
                charprocObject = self.reserveObject('charProc')
                self.outputStream(charprocObject, stream, extra=charprocDict)
                name = self._get_xobject_glyph_name(filename, charname)
                self.multi_byte_charprocs[name] = charprocObject
            cid_to_gid_map = ''.join(cid_to_gid_map).encode('utf-16be')
            self.outputStream(cidToGidMapObject, cid_to_gid_map)
            self.outputStream(toUnicodeMapObject, unicode_cmap)
            descriptor['MaxWidth'] = max_width
            self.writeObject(cidFontDictObject, cidFontDict)
            self.writeObject(type0FontDictObject, type0FontDict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(wObject, w)
            return type0FontDictObject
        ps_name = self._get_subsetted_psname(font.postscript_name, font.get_charmap())
        ps_name = ps_name.encode('ascii', 'replace')
        ps_name = Name(ps_name)
        pclt = font.get_sfnt_table('pclt') or {'capHeight': 0, 'xHeight': 0}
        post = font.get_sfnt_table('post') or {'italicAngle': (0, 0)}
        ff = font.face_flags
        sf = font.style_flags
        flags = 0
        symbolic = False
        if ff & FIXED_WIDTH:
            flags |= 1 << 0
        if 0:
            flags |= 1 << 1
        if symbolic:
            flags |= 1 << 2
        else:
            flags |= 1 << 5
        if sf & ITALIC:
            flags |= 1 << 6
        if 0:
            flags |= 1 << 16
        if 0:
            flags |= 1 << 17
        if 0:
            flags |= 1 << 18
        descriptor = {'Type': Name('FontDescriptor'), 'FontName': ps_name, 'Flags': flags, 'FontBBox': [cvt(x, nearest=False) for x in font.bbox], 'Ascent': cvt(font.ascender, nearest=False), 'Descent': cvt(font.descender, nearest=False), 'CapHeight': cvt(pclt['capHeight'], nearest=False), 'XHeight': cvt(pclt['xHeight']), 'ItalicAngle': post['italicAngle'][1], 'StemV': 0}
        if fonttype == 3:
            return embedTTFType3(font, characters, descriptor)
        elif fonttype == 42:
            return embedTTFType42(font, characters, descriptor)

    def alphaState(self, alpha):
        """Return name of an ExtGState that sets alpha to the given value."""
        state = self.alphaStates.get(alpha, None)
        if state is not None:
            return state[0]
        name = next(self._alpha_state_seq)
        self.alphaStates[alpha] = (name, {'Type': Name('ExtGState'), 'CA': alpha[0], 'ca': alpha[1]})
        return name

    def _soft_mask_state(self, smask):
        """
        Return an ExtGState that sets the soft mask to the given shading.

        Parameters
        ----------
        smask : Reference
            Reference to a shading in DeviceGray color space, whose luminosity
            is to be used as the alpha channel.

        Returns
        -------
        Name
        """
        state = self._soft_mask_states.get(smask, None)
        if state is not None:
            return state[0]
        name = next(self._soft_mask_seq)
        groupOb = self.reserveObject('transparency group for soft mask')
        self._soft_mask_states[smask] = (name, {'Type': Name('ExtGState'), 'AIS': False, 'SMask': {'Type': Name('Mask'), 'S': Name('Luminosity'), 'BC': [1], 'G': groupOb}})
        self._soft_mask_groups.append((groupOb, {'Type': Name('XObject'), 'Subtype': Name('Form'), 'FormType': 1, 'Group': {'S': Name('Transparency'), 'CS': Name('DeviceGray')}, 'Matrix': [1, 0, 0, 1, 0, 0], 'Resources': {'Shading': {'S': smask}}, 'BBox': [0, 0, 1, 1]}, [Name('S'), Op.shading]))
        return name

    def writeExtGSTates(self):
        self.writeObject(self._extGStateObject, dict([*self.alphaStates.values(), *self._soft_mask_states.values()]))

    def _write_soft_mask_groups(self):
        for ob, attributes, content in self._soft_mask_groups:
            self.beginStream(ob.id, None, attributes)
            self.output(*content)
            self.endStream()

    def hatchPattern(self, hatch_style):
        if hatch_style is not None:
            edge, face, hatch = hatch_style
            if edge is not None:
                edge = tuple(edge)
            if face is not None:
                face = tuple(face)
            hatch_style = (edge, face, hatch)
        pattern = self.hatchPatterns.get(hatch_style, None)
        if pattern is not None:
            return pattern
        name = next(self._hatch_pattern_seq)
        self.hatchPatterns[hatch_style] = name
        return name

    def writeHatches(self):
        hatchDict = dict()
        sidelen = 72.0
        for hatch_style, name in self.hatchPatterns.items():
            ob = self.reserveObject('hatch pattern')
            hatchDict[name] = ob
            res = {'Procsets': [Name(x) for x in 'PDF Text ImageB ImageC ImageI'.split()]}
            self.beginStream(ob.id, None, {'Type': Name('Pattern'), 'PatternType': 1, 'PaintType': 1, 'TilingType': 1, 'BBox': [0, 0, sidelen, sidelen], 'XStep': sidelen, 'YStep': sidelen, 'Resources': res, 'Matrix': [1, 0, 0, 1, 0, self.height * 72]})
            stroke_rgb, fill_rgb, hatch = hatch_style
            self.output(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2], Op.setrgb_stroke)
            if fill_rgb is not None:
                self.output(fill_rgb[0], fill_rgb[1], fill_rgb[2], Op.setrgb_nonstroke, 0, 0, sidelen, sidelen, Op.rectangle, Op.fill)
            self.output(mpl.rcParams['hatch.linewidth'], Op.setlinewidth)
            self.output(*self.pathOperations(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False))
            self.output(Op.fill_stroke)
            self.endStream()
        self.writeObject(self.hatchObject, hatchDict)

    def addGouraudTriangles(self, points, colors):
        """
        Add a Gouraud triangle shading.

        Parameters
        ----------
        points : np.ndarray
            Triangle vertices, shape (n, 3, 2)
            where n = number of triangles, 3 = vertices, 2 = x, y.
        colors : np.ndarray
            Vertex colors, shape (n, 3, 1) or (n, 3, 4)
            as with points, but last dimension is either (gray,)
            or (r, g, b, alpha).

        Returns
        -------
        Name, Reference
        """
        name = Name('GT%d' % len(self.gouraudTriangles))
        ob = self.reserveObject(f'Gouraud triangle {name}')
        self.gouraudTriangles.append((name, ob, points, colors))
        return (name, ob)

    def writeGouraudTriangles(self):
        gouraudDict = dict()
        for name, ob, points, colors in self.gouraudTriangles:
            gouraudDict[name] = ob
            shape = points.shape
            flat_points = points.reshape((shape[0] * shape[1], 2))
            colordim = colors.shape[2]
            assert colordim in (1, 4)
            flat_colors = colors.reshape((shape[0] * shape[1], colordim))
            if colordim == 4:
                colordim = 3
            points_min = np.min(flat_points, axis=0) - (1 << 8)
            points_max = np.max(flat_points, axis=0) + (1 << 8)
            factor = 4294967295 / (points_max - points_min)
            self.beginStream(ob.id, None, {'ShadingType': 4, 'BitsPerCoordinate': 32, 'BitsPerComponent': 8, 'BitsPerFlag': 8, 'ColorSpace': Name('DeviceRGB' if colordim == 3 else 'DeviceGray'), 'AntiAlias': False, 'Decode': [points_min[0], points_max[0], points_min[1], points_max[1]] + [0, 1] * colordim})
            streamarr = np.empty((shape[0] * shape[1],), dtype=[('flags', 'u1'), ('points', '>u4', (2,)), ('colors', 'u1', (colordim,))])
            streamarr['flags'] = 0
            streamarr['points'] = (flat_points - points_min) * factor
            streamarr['colors'] = flat_colors[:, :colordim] * 255.0
            self.write(streamarr.tobytes())
            self.endStream()
        self.writeObject(self.gouraudObject, gouraudDict)

    def imageObject(self, image):
        """Return name of an image XObject representing the given image."""
        entry = self._images.get(id(image), None)
        if entry is not None:
            return entry[1]
        name = next(self._image_seq)
        ob = self.reserveObject(f'image {name}')
        self._images[id(image)] = (image, name, ob)
        return name

    def _unpack(self, im):
        """
        Unpack image array *im* into ``(data, alpha)``, which have shape
        ``(height, width, 3)`` (RGB) or ``(height, width, 1)`` (grayscale or
        alpha), except that alpha is None if the image is fully opaque.
        """
        im = im[::-1]
        if im.ndim == 2:
            return (im, None)
        else:
            rgb = im[:, :, :3]
            rgb = np.array(rgb, order='C')
            if im.shape[2] == 4:
                alpha = im[:, :, 3][..., None]
                if np.all(alpha == 255):
                    alpha = None
                else:
                    alpha = np.array(alpha, order='C')
            else:
                alpha = None
            return (rgb, alpha)

    def _writePng(self, img):
        """
        Write the image *img* into the pdf file using png
        predictors with Flate compression.
        """
        buffer = BytesIO()
        img.save(buffer, format='png')
        buffer.seek(8)
        png_data = b''
        bit_depth = palette = None
        while True:
            length, type = struct.unpack(b'!L4s', buffer.read(8))
            if type in [b'IHDR', b'PLTE', b'IDAT']:
                data = buffer.read(length)
                if len(data) != length:
                    raise RuntimeError('truncated data')
                if type == b'IHDR':
                    bit_depth = int(data[8])
                elif type == b'PLTE':
                    palette = data
                elif type == b'IDAT':
                    png_data += data
            elif type == b'IEND':
                break
            else:
                buffer.seek(length, 1)
            buffer.seek(4, 1)
        return (png_data, bit_depth, palette)

    def _writeImg(self, data, id, smask=None):
        """
        Write the image *data*, of shape ``(height, width, 1)`` (grayscale) or
        ``(height, width, 3)`` (RGB), as pdf object *id* and with the soft mask
        (alpha channel) *smask*, which should be either None or a ``(height,
        width, 1)`` array.
        """
        height, width, color_channels = data.shape
        obj = {'Type': Name('XObject'), 'Subtype': Name('Image'), 'Width': width, 'Height': height, 'ColorSpace': Name({1: 'DeviceGray', 3: 'DeviceRGB'}[color_channels]), 'BitsPerComponent': 8}
        if smask:
            obj['SMask'] = smask
        if mpl.rcParams['pdf.compression']:
            if data.shape[-1] == 1:
                data = data.squeeze(axis=-1)
            png = {'Predictor': 10, 'Colors': color_channels, 'Columns': width}
            img = Image.fromarray(data)
            img_colors = img.getcolors(maxcolors=256)
            if color_channels == 3 and img_colors is not None:
                num_colors = len(img_colors)
                palette = np.array([comp for _, color in img_colors for comp in color], dtype=np.uint8)
                palette24 = palette[0::3].astype(np.uint32) << 16 | palette[1::3].astype(np.uint32) << 8 | palette[2::3]
                rgb24 = data[:, :, 0].astype(np.uint32) << 16 | data[:, :, 1].astype(np.uint32) << 8 | data[:, :, 2]
                indices = np.argsort(palette24).astype(np.uint8)
                rgb8 = indices[np.searchsorted(palette24, rgb24, sorter=indices)]
                img = Image.fromarray(rgb8, mode='P')
                img.putpalette(palette)
                png_data, bit_depth, palette = self._writePng(img)
                if bit_depth is None or palette is None:
                    raise RuntimeError('invalid PNG header')
                palette = palette[:num_colors * 3]
                obj['ColorSpace'] = [Name('Indexed'), Name('DeviceRGB'), num_colors - 1, palette]
                obj['BitsPerComponent'] = bit_depth
                png['Colors'] = 1
                png['BitsPerComponent'] = bit_depth
            else:
                png_data, _, _ = self._writePng(img)
        else:
            png = None
        self.beginStream(id, self.reserveObject('length of image stream'), obj, png=png)
        if png:
            self.currentstream.write(png_data)
        else:
            self.currentstream.write(data.tobytes())
        self.endStream()

    def writeImages(self):
        for img, name, ob in self._images.values():
            data, adata = self._unpack(img)
            if adata is not None:
                smaskObject = self.reserveObject('smask')
                self._writeImg(adata, smaskObject.id)
            else:
                smaskObject = None
            self._writeImg(data, ob.id, smaskObject)

    def markerObject(self, path, trans, fill, stroke, lw, joinstyle, capstyle):
        """Return name of a marker XObject representing the given path."""
        pathops = self.pathOperations(path, trans, simplify=False)
        key = (tuple(pathops), bool(fill), bool(stroke), joinstyle, capstyle)
        result = self.markers.get(key)
        if result is None:
            name = Name('M%d' % len(self.markers))
            ob = self.reserveObject('marker %d' % len(self.markers))
            bbox = path.get_extents(trans)
            self.markers[key] = [name, ob, bbox, lw]
        else:
            if result[-1] < lw:
                result[-1] = lw
            name = result[0]
        return name

    def writeMarkers(self):
        for (pathops, fill, stroke, joinstyle, capstyle), (name, ob, bbox, lw) in self.markers.items():
            bbox = bbox.padded(lw * 5)
            self.beginStream(ob.id, None, {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': list(bbox.extents)})
            self.output(GraphicsContextPdf.joinstyles[joinstyle], Op.setlinejoin)
            self.output(GraphicsContextPdf.capstyles[capstyle], Op.setlinecap)
            self.output(*pathops)
            self.output(Op.paint_path(fill, stroke))
            self.endStream()

    def pathCollectionObject(self, gc, path, trans, padding, filled, stroked):
        name = Name('P%d' % len(self.paths))
        ob = self.reserveObject('path %d' % len(self.paths))
        self.paths.append((name, path, trans, ob, gc.get_joinstyle(), gc.get_capstyle(), padding, filled, stroked))
        return name

    def writePathCollectionTemplates(self):
        for name, path, trans, ob, joinstyle, capstyle, padding, filled, stroked in self.paths:
            pathops = self.pathOperations(path, trans, simplify=False)
            bbox = path.get_extents(trans)
            if not np.all(np.isfinite(bbox.extents)):
                extents = [0, 0, 0, 0]
            else:
                bbox = bbox.padded(padding)
                extents = list(bbox.extents)
            self.beginStream(ob.id, None, {'Type': Name('XObject'), 'Subtype': Name('Form'), 'BBox': extents})
            self.output(GraphicsContextPdf.joinstyles[joinstyle], Op.setlinejoin)
            self.output(GraphicsContextPdf.capstyles[capstyle], Op.setlinecap)
            self.output(*pathops)
            self.output(Op.paint_path(filled, stroked))
            self.endStream()

    @staticmethod
    def pathOperations(path, transform, clip=None, simplify=None, sketch=None):
        return [Verbatim(_path.convert_to_string(path, transform, clip, simplify, sketch, 6, [Op.moveto.value, Op.lineto.value, b'', Op.curveto.value, Op.closepath.value], True))]

    def writePath(self, path, transform, clip=False, sketch=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72, self.height * 72)
            simplify = path.should_simplify
        else:
            clip = None
            simplify = False
        cmds = self.pathOperations(path, transform, clip, simplify=simplify, sketch=sketch)
        self.output(*cmds)

    def reserveObject(self, name=''):
        """
        Reserve an ID for an indirect object.

        The name is used for debugging in case we forget to print out
        the object with writeObject.
        """
        id = next(self._object_seq)
        self.xrefTable.append([None, 0, name])
        return Reference(id)

    def recordXref(self, id):
        self.xrefTable[id][0] = self.fh.tell() - self.tell_base

    def writeObject(self, object, contents):
        self.recordXref(object.id)
        object.write(contents, self)

    def writeXref(self):
        """Write out the xref table."""
        self.startxref = self.fh.tell() - self.tell_base
        self.write(b'xref\n0 %d\n' % len(self.xrefTable))
        for i, (offset, generation, name) in enumerate(self.xrefTable):
            if offset is None:
                raise AssertionError('No offset for object %d (%s)' % (i, name))
            else:
                key = b'f' if name == 'the zero object' else b'n'
                text = b'%010d %05d %b \n' % (offset, generation, key)
                self.write(text)

    def writeInfoDict(self):
        """Write out the info dictionary, checking it for good form"""
        self.infoObject = self.reserveObject('info')
        self.writeObject(self.infoObject, self.infoDict)

    def writeTrailer(self):
        """Write out the PDF trailer."""
        self.write(b'trailer\n')
        self.write(pdfRepr({'Size': len(self.xrefTable), 'Root': self.rootObject, 'Info': self.infoObject}))
        self.write(b'\nstartxref\n%d\n%%%%EOF\n' % self.startxref)