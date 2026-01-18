import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
class FreeTypeFace:
    """FreeType typographic face object.

    Keeps the reference count to the face at +1 as long as this object exists. If other objects
    want to keep a face without a reference to this object, they should increase the reference
    counter themselves and decrease it again when done.
    """

    def __init__(self, ft_face):
        assert ft_face is not None
        self.ft_face = ft_face
        self._get_best_name()

    @classmethod
    def from_file(cls, file_name):
        ft_library = ft_get_library()
        ft_face = FT_Face()
        FT_New_Face(ft_library, asbytes(file_name), 0, byref(ft_face))
        return cls(ft_face)

    @classmethod
    def from_fontconfig(cls, match):
        if match.face is not None:
            FT_Reference_Face(match.face)
            return cls(match.face)
        else:
            if not match.file:
                raise base.FontException(f'No filename for "{match.name}"')
            return cls.from_file(match.file)

    @property
    def name(self):
        return self._name

    @property
    def family_name(self):
        return asstr(self.ft_face.contents.family_name)

    @property
    def style_flags(self):
        return self.ft_face.contents.style_flags

    @property
    def bold(self):
        return self.style_flags & FT_STYLE_FLAG_BOLD != 0

    @property
    def italic(self):
        return self.style_flags & FT_STYLE_FLAG_ITALIC != 0

    @property
    def face_flags(self):
        return self.ft_face.contents.face_flags

    def __del__(self):
        if self.ft_face is not None:
            FT_Done_Face(self.ft_face)
            self.ft_face = None

    def set_char_size(self, size, dpi):
        face_size = float_to_f26p6(size)
        try:
            FT_Set_Char_Size(self.ft_face, 0, face_size, dpi, dpi)
            return True
        except FreeTypeError as e:
            if e.errcode == 23:
                return False
            else:
                raise

    def get_character_index(self, character):
        return get_fontconfig().char_index(self.ft_face, character)

    def get_glyph_slot(self, glyph_index):
        FT_Load_Glyph(self.ft_face, glyph_index, FT_LOAD_RENDER)
        return self.ft_face.contents.glyph.contents

    def get_font_metrics(self, size, dpi):
        if self.set_char_size(size, dpi):
            metrics = self.ft_face.contents.size.contents.metrics
            if metrics.ascender == 0 and metrics.descender == 0:
                return self._get_font_metrics_workaround()
            else:
                return FreeTypeFontMetrics(ascent=int(f26p6_to_float(metrics.ascender)), descent=int(f26p6_to_float(metrics.descender)))
        else:
            return self._get_font_metrics_workaround()

    def _get_font_metrics_workaround(self):
        i = self.get_character_index('X')
        self.get_glyph_slot(i)
        ascent = self.ft_face.contents.available_sizes.contents.height
        return FreeTypeFontMetrics(ascent=ascent, descent=-ascent // 4)

    def _get_best_name(self):
        self._name = asstr(self.ft_face.contents.family_name)
        self._get_font_family_from_ttf

    def _get_font_family_from_ttf(self):
        return
        if self.face_flags & FT_FACE_FLAG_SFNT:
            name = FT_SfntName()
            for i in range(FT_Get_Sfnt_Name_Count(self.ft_face)):
                try:
                    FT_Get_Sfnt_Name(self.ft_face, i, name)
                    if not (name.platform_id == TT_PLATFORM_MICROSOFT and name.encoding_id == TT_MS_ID_UNICODE_CS):
                        continue
                    self._name = name.string.decode('utf-16be', 'ignore')
                except:
                    continue