import io
import math
import os
import typing
import weakref
def _get_font_properties(doc: fitz.Document, xref: int) -> tuple:
    fontname, ext, stype, buffer = doc.extract_font(xref)
    asc = 0.8
    dsc = -0.2
    if ext == '':
        return (fontname, ext, stype, asc, dsc)
    if buffer:
        try:
            font = fitz.Font(fontbuffer=buffer)
            asc = font.ascender
            dsc = font.descender
            bbox = font.bbox
            if asc - dsc < 1:
                if bbox.y0 < dsc:
                    dsc = bbox.y0
                asc = 1 - dsc
        except Exception:
            fitz.exception_info()
            asc *= 1.2
            dsc *= 1.2
        return (fontname, ext, stype, asc, dsc)
    if ext != 'n/a':
        try:
            font = fitz.Font(fontname)
            asc = font.ascender
            dsc = font.descender
        except Exception:
            fitz.exception_info()
            asc *= 1.2
            dsc *= 1.2
    else:
        asc *= 1.2
        dsc *= 1.2
    return (fontname, ext, stype, asc, dsc)