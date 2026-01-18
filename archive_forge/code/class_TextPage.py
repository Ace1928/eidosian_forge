import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class TextPage:

    def __init__(self, *args):
        if args_match(args, mupdf.FzRect):
            mediabox = args[0]
            self.this = mupdf.FzStextPage(mediabox)
        elif args_match(args, mupdf.FzStextPage):
            self.this = args[0]
        else:
            raise Exception(f'Unrecognised args: {args}')
        self.thisown = True
        self.parent = None

    def _extractText(self, format_):
        this_tpage = self.this
        res = mupdf.fz_new_buffer(1024)
        out = mupdf.FzOutput(res)
        if format_ == 1:
            mupdf.fz_print_stext_page_as_html(out, this_tpage, 0)
        elif format_ == 3:
            mupdf.fz_print_stext_page_as_xml(out, this_tpage, 0)
        elif format_ == 4:
            mupdf.fz_print_stext_page_as_xhtml(out, this_tpage, 0)
        else:
            JM_print_stext_page_as_text(res, this_tpage)
        out.fz_close_output()
        text = JM_EscapeStrFromBuffer(res)
        return text

    def _getNewBlockList(self, page_dict, raw):
        JM_make_textpage_dict(self.this, page_dict, raw)

    def _textpage_dict(self, raw=False):
        page_dict = {'width': self.rect.width, 'height': self.rect.height}
        self._getNewBlockList(page_dict, raw)
        return page_dict

    def extractBLOCKS(self):
        """Return a list with text block information."""
        if g_use_extra:
            return extra.extractBLOCKS(self.this)
        block_n = -1
        this_tpage = self.this
        tp_rect = mupdf.FzRect(this_tpage.m_internal.mediabox)
        res = mupdf.fz_new_buffer(1024)
        lines = []
        for block in this_tpage:
            block_n += 1
            blockrect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
            if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_TEXT:
                mupdf.fz_clear_buffer(res)
                line_n = -1
                last_char = 0
                for line in block:
                    line_n += 1
                    linerect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
                    for ch in line:
                        cbbox = JM_char_bbox(line, ch)
                        if not JM_rects_overlap(tp_rect, cbbox) and (not mupdf.fz_is_infinite_rect(tp_rect)):
                            continue
                        JM_append_rune(res, ch.m_internal.c)
                        last_char = ch.m_internal.c
                        linerect = mupdf.fz_union_rect(linerect, cbbox)
                    if last_char != 10 and (not mupdf.fz_is_empty_rect(linerect)):
                        mupdf.fz_append_byte(res, 10)
                    blockrect = mupdf.fz_union_rect(blockrect, linerect)
                text = JM_EscapeStrFromBuffer(res)
            elif JM_rects_overlap(tp_rect, block.m_internal.bbox) or mupdf.fz_is_infinite_rect(tp_rect):
                img = block.i_image()
                cs = img.colorspace()
                text = '<image: %s, width: %d, height: %d, bpc: %d>' % (mupdf.fz_colorspace_name(cs), img.w(), img.h(), img.bpc())
                blockrect = mupdf.fz_union_rect(blockrect, mupdf.FzRect(block.m_internal.bbox))
            if not mupdf.fz_is_empty_rect(blockrect):
                litem = (blockrect.x0, blockrect.y0, blockrect.x1, blockrect.y1, text, block_n, block.m_internal.type)
                lines.append(litem)
        return lines

    def extractDICT(self, cb=None, sort=False) -> dict:
        """Return page content as a Python dict of images and text spans."""
        val = self._textpage_dict(raw=False)
        if cb is not None:
            val['width'] = cb.width
            val['height'] = cb.height
        if sort is True:
            blocks = val['blocks']
            blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
            val['blocks'] = blocks
        return val

    def extractHTML(self) -> str:
        """Return page content as a HTML string."""
        return self._extractText(1)

    def extractIMGINFO(self, hashes=0):
        """Return a list with image meta information."""
        block_n = -1
        this_tpage = self.this
        rc = []
        for block in this_tpage:
            block_n += 1
            if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_TEXT:
                continue
            img = block.i_image()
            img_size = 0
            if mupdf_version_tuple >= (1, 24):
                compr_buff = mupdf.fz_compressed_image_buffer(img)
                if compr_buff:
                    img_size = compr_buff.fz_compressed_buffer_size()
                    compr_buff = None
            else:
                compr_buff = mupdf.ll_fz_compressed_image_buffer(img.m_internal)
                if compr_buff:
                    img_size = mupdf.ll_fz_compressed_buffer_size(compr_buff)
            if hashes:
                r = mupdf.FzIrect(FZ_MIN_INF_RECT, FZ_MIN_INF_RECT, FZ_MAX_INF_RECT, FZ_MAX_INF_RECT)
                assert mupdf.fz_is_infinite_irect(r)
                m = mupdf.FzMatrix(img.w(), 0, 0, img.h(), 0, 0)
                pix, w, h = mupdf.fz_get_pixmap_from_image(img, r, m)
                digest = mupdf.fz_md5_pixmap2(pix)
                digest = bytes(digest)
                if img_size == 0:
                    img_size = img.w() * img.h() * img.n()
            cs = mupdf.FzColorspace(mupdf.ll_fz_keep_colorspace(img.m_internal.colorspace))
            block_dict = dict()
            block_dict[dictkey_number] = block_n
            block_dict[dictkey_bbox] = JM_py_from_rect(block.m_internal.bbox)
            block_dict[dictkey_matrix] = JM_py_from_matrix(block.i_transform())
            block_dict[dictkey_width] = img.w()
            block_dict[dictkey_height] = img.h()
            block_dict[dictkey_colorspace] = mupdf.fz_colorspace_n(cs)
            block_dict[dictkey_cs_name] = mupdf.fz_colorspace_name(cs)
            block_dict[dictkey_xres] = img.xres()
            block_dict[dictkey_yres] = img.yres()
            block_dict[dictkey_bpc] = img.bpc()
            block_dict[dictkey_size] = img_size
            if hashes:
                block_dict['digest'] = digest
            rc.append(block_dict)
        return rc

    def extractJSON(self, cb=None, sort=False) -> str:
        """Return 'extractDICT' converted to JSON format."""
        import base64
        import json
        val = self._textpage_dict(raw=False)

        class b64encode(json.JSONEncoder):

            def default(self, s):
                if type(s) in (bytes, bytearray):
                    return base64.b64encode(s).decode()
        if cb is not None:
            val['width'] = cb.width
            val['height'] = cb.height
        if sort is True:
            blocks = val['blocks']
            blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
            val['blocks'] = blocks
        val = json.dumps(val, separators=(',', ':'), cls=b64encode, indent=1)
        return val

    def extractRAWDICT(self, cb=None, sort=False) -> dict:
        """Return page content as a Python dict of images and text characters."""
        val = self._textpage_dict(raw=True)
        if cb is not None:
            val['width'] = cb.width
            val['height'] = cb.height
        if sort is True:
            blocks = val['blocks']
            blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
            val['blocks'] = blocks
        return val

    def extractRAWJSON(self, cb=None, sort=False) -> str:
        """Return 'extractRAWDICT' converted to JSON format."""
        import base64
        import json
        val = self._textpage_dict(raw=True)

        class b64encode(json.JSONEncoder):

            def default(self, s):
                if type(s) in (bytes, bytearray):
                    return base64.b64encode(s).decode()
        if cb is not None:
            val['width'] = cb.width
            val['height'] = cb.height
        if sort is True:
            blocks = val['blocks']
            blocks.sort(key=lambda b: (b['bbox'][3], b['bbox'][0]))
            val['blocks'] = blocks
        val = json.dumps(val, separators=(',', ':'), cls=b64encode, indent=1)
        return val

    def extractSelection(self, pointa, pointb):
        a = JM_point_from_py(pointa)
        b = JM_point_from_py(pointb)
        found = mupdf.fz_copy_selection(self.this, a, b, 0)
        return found

    def extractText(self, sort=False) -> str:
        """Return simple, bare text on the page."""
        if sort is False:
            return self._extractText(0)
        blocks = self.extractBLOCKS()[:]
        blocks.sort(key=lambda b: (b[3], b[0]))
        return ''.join([b[4] for b in blocks])

    def extractTextbox(self, rect):
        this_tpage = self.this
        assert isinstance(this_tpage, mupdf.FzStextPage)
        area = JM_rect_from_py(rect)
        found = JM_copy_rectangle(this_tpage, area)
        rc = PyUnicode_DecodeRawUnicodeEscape(found)
        return rc

    def extractWORDS(self, delimiters=None):
        """Return a list with text word information."""
        if g_use_extra:
            return extra.extractWORDS(self.this, delimiters)
        buflen = 0
        block_n = -1
        wbbox = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
        this_tpage = self.this
        tp_rect = mupdf.FzRect(this_tpage.m_internal.mediabox)
        lines = None
        buff = mupdf.fz_new_buffer(64)
        lines = []
        for block in this_tpage:
            block_n += 1
            if block.m_internal.type != mupdf.FZ_STEXT_BLOCK_TEXT:
                continue
            line_n = -1
            for line in block:
                line_n += 1
                word_n = 0
                mupdf.fz_clear_buffer(buff)
                buflen = 0
                for ch in line:
                    cbbox = JM_char_bbox(line, ch)
                    if not JM_rects_overlap(tp_rect, cbbox) and (not mupdf.fz_is_infinite_rect(tp_rect)):
                        continue
                    word_delimiter = JM_is_word_delimiter(ch.m_internal.c, delimiters)
                    if word_delimiter:
                        if buflen == 0:
                            continue
                        if not mupdf.fz_is_empty_rect(wbbox):
                            word_n, wbbox = JM_append_word(lines, buff, wbbox, block_n, line_n, word_n)
                        mupdf.fz_clear_buffer(buff)
                        buflen = 0
                        continue
                    JM_append_rune(buff, ch.m_internal.c)
                    buflen += 1
                    wbbox = mupdf.fz_union_rect(wbbox, JM_char_bbox(line, ch))
                if buflen and (not mupdf.fz_is_empty_rect(wbbox)):
                    word_n, wbbox = JM_append_word(lines, buff, wbbox, block_n, line_n, word_n)
                buflen = 0
        return lines

    def extractXHTML(self) -> str:
        """Return page content as a XHTML string."""
        return self._extractText(4)

    def extractXML(self) -> str:
        """Return page content as a XML string."""
        return self._extractText(3)

    def poolsize(self):
        """TextPage current poolsize."""
        tpage = self.this
        pool = mupdf.Pool(tpage.m_internal.pool)
        size = mupdf.fz_pool_size(pool)
        pool.m_internal = None
        return size

    @property
    def rect(self):
        """Page rectangle."""
        this_tpage = self.this
        mediabox = this_tpage.m_internal.mediabox
        val = JM_py_from_rect(mediabox)
        val = Rect(val)
        return val

    def search(self, needle, hit_max=0, quads=1):
        """Locate 'needle' returning rects or quads."""
        val = JM_search_stext_page(self.this, needle)
        if not val:
            return val
        items = len(val)
        for i in range(items):
            q = Quad(val[i])
            if quads:
                val[i] = q
            else:
                val[i] = q.rect
        if quads:
            return val
        i = 0
        while i < items - 1:
            v1 = val[i]
            v2 = val[i + 1]
            if v1.y1 != v2.y1 or (v1 & v2).is_empty:
                i += 1
                continue
            val[i] = v1 | v2
            del val[i + 1]
            items -= 1
        return val
    extractTEXT = extractText