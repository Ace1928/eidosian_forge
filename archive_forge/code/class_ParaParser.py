import re
import sys
import copy
import unicodedata
import reportlab.lib.sequencer
from reportlab.lib.abag import ABag
from reportlab.lib.utils import ImageReader, annotateException, encode_label, asUnicode
from reportlab.lib.colors import toColor, black
from reportlab.lib.fonts import tt2ps, ps2tt
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch,mm,cm,pica
from reportlab.rl_config import platypus_link_underline
from html.parser import HTMLParser
from html.entities import name2codepoint
class ParaParser(HTMLParser):

    def __getattr__(self, attrName):
        """This way we can handle <TAG> the same way as <tag> (ignoring case)."""
        if attrName != attrName.lower() and attrName != 'caseSensitive' and (not self.caseSensitive) and (attrName.startswith('start_') or attrName.startswith('end_')):
            return getattr(self, attrName.lower())
        raise AttributeError(attrName)

    def start_b(self, attributes):
        self._push('b', bold=1)

    def end_b(self):
        self._pop('b')

    def start_strong(self, attributes):
        self._push('strong', bold=1)

    def end_strong(self):
        self._pop('strong')

    def start_i(self, attributes):
        self._push('i', italic=1)

    def end_i(self):
        self._pop('i')

    def start_em(self, attributes):
        self._push('em', italic=1)

    def end_em(self):
        self._pop('em')

    def _new_line(self, k):
        frag = self._stack[-1]
        frag.us_lines = frag.us_lines + [(self.nlines, k, getattr(frag, k + 'Color', self._defaultLineColors[k]), getattr(frag, k + 'Width', self._defaultLineWidths[k]), getattr(frag, k + 'Offset', self._defaultLineOffsets[k]), frag.rise, _lineRepeats[getattr(frag, k + 'Kind', 'single')], getattr(frag, k + 'Gap', self._defaultLineGaps[k]))]
        self.nlines += 1

    def start_u(self, attributes):
        A = self.getAttributes(attributes, _uAttrMap)
        self._push('u', **A)
        self._new_line('underline')

    def end_u(self):
        self._pop('u')

    def start_strike(self, attributes):
        A = self.getAttributes(attributes, _strikeAttrMap)
        self._push('strike', strike=1, **A)
        self._new_line('strike')

    def end_strike(self):
        self._pop('strike')

    def _handle_link(self, tag, attributes):
        A = self.getAttributes(attributes, _linkAttrMap)
        underline = A.pop('underline', self._defaultLinkUnderline)
        A['link'] = self._stack[-1].link + [(self.nlinks, A.pop('link', '').strip())]
        self.nlinks += 1
        self._push(tag, **A)
        if underline:
            self._new_line('underline')

    def start_link(self, attributes):
        self._handle_link('link', attributes)

    def end_link(self):
        if self._pop('link').link is None:
            raise ValueError('<link> has no target or href')

    def start_a(self, attributes):
        anchor = 'name' in attributes
        if anchor:
            A = self.getAttributes(attributes, _anchorAttrMap)
            name = A.get('name', None)
            name = name.strip()
            if not name:
                self._syntax_error('<a name="..."/> anchor variant requires non-blank name')
            if len(A) > 1:
                self._syntax_error('<a name="..."/> anchor variant only allows name attribute')
                A = dict(name=A['name'])
            A['_selfClosingTag'] = 'anchor'
            self._push('a', **A)
        else:
            self._handle_link('a', attributes)

    def end_a(self):
        frag = self._stack[-1]
        sct = getattr(frag, '_selfClosingTag', '')
        if sct:
            if not (sct == 'anchor' and frag.name):
                raise ValueError('Parser failure in <a/>')
            defn = frag.cbDefn = ABag()
            defn.label = defn.kind = 'anchor'
            defn.name = frag.name
            del frag.name, frag._selfClosingTag
            self.handle_data('')
            self._pop('a')
        elif self._pop('a').link is None:
            raise ValueError('<link> has no href')

    def start_img(self, attributes):
        A = self.getAttributes(attributes, _imgAttrMap)
        if not A.get('src'):
            self._syntax_error('<img> needs src attribute')
        A['_selfClosingTag'] = 'img'
        self._push('img', **A)

    def end_img(self):
        frag = self._stack[-1]
        if not getattr(frag, '_selfClosingTag', ''):
            raise ValueError('Parser failure in <img/>')
        defn = frag.cbDefn = ABag()
        defn.kind = 'img'
        defn.src = getattr(frag, 'src', None)
        defn.image = ImageReader(defn.src)
        size = defn.image.getSize()
        defn.width = getattr(frag, 'width', size[0])
        defn.height = getattr(frag, 'height', size[1])
        defn.valign = getattr(frag, 'valign', 'bottom')
        del frag._selfClosingTag
        self.handle_data('')
        self._pop('img')

    def start_super(self, attributes):
        A = self.getAttributes(attributes, _supAttrMap)
        self._push('super', **A)
        frag = self._stack[-1]
        frag.rise += fontSizeNormalize(frag, 'supr', frag.fontSize * supFraction)
        frag.fontSize = fontSizeNormalize(frag, 'sups', frag.fontSize - min(sizeDelta, 0.2 * frag.fontSize))

    def end_super(self):
        self._pop('super')
    start_sup = start_super
    end_sup = end_super

    def start_sub(self, attributes):
        A = self.getAttributes(attributes, _supAttrMap)
        self._push('sub', **A)
        frag = self._stack[-1]
        frag.rise -= fontSizeNormalize(frag, 'supr', frag.fontSize * subFraction)
        frag.fontSize = fontSizeNormalize(frag, 'sups', frag.fontSize - min(sizeDelta, 0.2 * frag.fontSize))

    def end_sub(self):
        self._pop('sub')

    def start_nobr(self, attrs):
        self.getAttributes(attrs, {})
        self._push('nobr', nobr=True)

    def end_nobr(self):
        self._pop('nobr')

    def handle_charref(self, name):
        try:
            if name[0] == 'x':
                n = int(name[1:], 16)
            else:
                n = int(name)
        except ValueError:
            self.unknown_charref(name)
            return
        self.handle_data(chr(n))

    def syntax_error(self, lineno, message):
        self._syntax_error(message)

    def _syntax_error(self, message):
        if message[:10] == 'attribute ' and message[-17:] == ' value not quoted':
            return
        if self._crashOnError:
            raise ValueError('paraparser: syntax error: %s' % message)
        self.errors.append(message)

    def start_greek(self, attr):
        self._push('greek', greek=1)

    def end_greek(self):
        self._pop('greek')

    def start_unichar(self, attr):
        if 'name' in attr:
            if 'code' in attr:
                self._syntax_error('<unichar/> invalid with both name and code attributes')
            try:
                v = unicodedata.lookup(attr['name'])
            except KeyError:
                self._syntax_error('<unichar/> invalid name attribute\n"%s"' % ascii(attr['name']))
                v = '\x00'
        elif 'code' in attr:
            try:
                v = attr['code'].lower()
                if v.startswith('0x'):
                    v = int(v, 16)
                else:
                    v = int(v, 0)
                v = chr(v)
            except:
                self._syntax_error('<unichar/> invalid code attribute %s' % ascii(attr['code']))
                v = '\x00'
        else:
            v = None
            if attr:
                self._syntax_error('<unichar/> invalid attribute %s' % list(attr.keys())[0])
        if v is not None:
            self.handle_data(v)
        self._push('unichar', _selfClosingTag='unichar')

    def end_unichar(self):
        self._pop('unichar')

    def start_font(self, attr):
        A = self.getAttributes(attr, _spanAttrMap)
        if 'fontName' in A:
            A['fontName'], A['bold'], A['italic'] = ps2tt(A['fontName'])
        self._push('font', **A)

    def end_font(self):
        self._pop('font')

    def start_span(self, attr):
        A = self.getAttributes(attr, _spanAttrMap)
        if 'style' in A:
            style = self.findSpanStyle(A.pop('style'))
            D = {}
            for k in 'fontName fontSize textColor backColor'.split():
                v = getattr(style, k, self)
                if v is self:
                    continue
                D[k] = v
            D.update(A)
            A = D
        if 'fontName' in A:
            A['fontName'], A['bold'], A['italic'] = ps2tt(A['fontName'])
        self._push('span', **A)

    def end_span(self):
        self._pop('span')

    def start_br(self, attr):
        self._push('br', _selfClosingTag='br', lineBreak=True, text='')

    def end_br(self):
        frag = self._stack[-1]
        if not (frag._selfClosingTag == 'br' and frag.lineBreak):
            raise ValueError('Parser failure in <br/>')
        del frag._selfClosingTag
        self.handle_data('')
        self._pop('br')

    def _initial_frag(self, attr, attrMap, bullet=0):
        style = self._style
        if attr != {}:
            style = copy.deepcopy(style)
            _applyAttributes(style, self.getAttributes(attr, attrMap))
            self._style = style
        frag = ParaFrag()
        frag.rise = 0
        frag.greek = 0
        frag.link = []
        try:
            if bullet:
                frag.fontName, frag.bold, frag.italic = ps2tt(style.bulletFontName)
                frag.fontSize = style.bulletFontSize
                frag.textColor = hasattr(style, 'bulletColor') and style.bulletColor or style.textColor
            else:
                frag.fontName, frag.bold, frag.italic = ps2tt(style.fontName)
                frag.fontSize = style.fontSize
                frag.textColor = style.textColor
        except:
            annotateException('error with style name=%s' % style.name)
        frag.us_lines = []
        self.nlinks = self.nlines = 0
        self._defaultLineWidths = dict(underline=getattr(style, 'underlineWidth', ''), strike=getattr(style, 'strikeWidth', ''))
        self._defaultLineColors = dict(underline=getattr(style, 'underlineColor', ''), strike=getattr(style, 'strikeColor', ''))
        self._defaultLineOffsets = dict(underline=getattr(style, 'underlineOffset', ''), strike=getattr(style, 'strikeOffset', ''))
        self._defaultLineGaps = dict(underline=getattr(style, 'underlineGap', ''), strike=getattr(style, 'strikeGap', ''))
        self._defaultLinkUnderline = getattr(style, 'linkUnderline', platypus_link_underline)
        return frag

    def start_para(self, attr):
        frag = self._initial_frag(attr, _paraAttrMap)
        frag.__tag__ = 'para'
        self._stack = [frag]

    def end_para(self):
        self._pop('para')

    def start_bullet(self, attr):
        if hasattr(self, 'bFragList'):
            self._syntax_error('only one <bullet> tag allowed')
        self.bFragList = []
        frag = self._initial_frag(attr, _bulletAttrMap, 1)
        frag.isBullet = 1
        frag.__tag__ = 'bullet'
        self._stack.append(frag)

    def end_bullet(self):
        self._pop('bullet')

    def start_seqdefault(self, attr):
        try:
            default = attr['id']
        except KeyError:
            default = None
        self._seq.setDefaultCounter(default)

    def end_seqdefault(self):
        pass

    def start_seqreset(self, attr):
        try:
            id = attr['id']
        except KeyError:
            id = None
        try:
            base = int(attr['base'])
        except:
            base = 0
        self._seq.reset(id, base)

    def end_seqreset(self):
        pass

    def start_seqchain(self, attr):
        try:
            order = attr['order']
        except KeyError:
            order = ''
        order = order.split()
        seq = self._seq
        for p, c in zip(order[:-1], order[1:]):
            seq.chain(p, c)
    end_seqchain = end_seqreset

    def start_seqformat(self, attr):
        try:
            id = attr['id']
        except KeyError:
            id = None
        try:
            value = attr['value']
        except KeyError:
            value = '1'
        self._seq.setFormat(id, value)
    end_seqformat = end_seqreset
    start_seqDefault = start_seqdefault
    end_seqDefault = end_seqdefault
    start_seqReset = start_seqreset
    end_seqReset = end_seqreset
    start_seqChain = start_seqchain
    end_seqChain = end_seqchain
    start_seqFormat = start_seqformat
    end_seqFormat = end_seqformat

    def start_seq(self, attr):
        if 'template' in attr:
            templ = attr['template']
            self.handle_data(templ % self._seq)
            return
        elif 'id' in attr:
            id = attr['id']
        else:
            id = None
        increment = attr.get('inc', None)
        if not increment:
            output = self._seq.nextf(id)
        elif increment.lower() == 'no':
            output = self._seq.thisf(id)
        else:
            incr = int(increment)
            output = self._seq.thisf(id)
            self._seq.reset(id, self._seq._this() + incr)
        self.handle_data(output)

    def end_seq(self):
        pass

    def start_ondraw(self, attr):
        defn = ABag()
        if 'name' in attr:
            defn.name = attr['name']
        else:
            self._syntax_error('<onDraw> needs at least a name attribute')
        defn.label = attr.get('label', None)
        defn.kind = 'onDraw'
        self._push('ondraw', cbDefn=defn)
        self.handle_data('')
        self._pop('ondraw')
    start_onDraw = start_ondraw
    end_onDraw = end_ondraw = end_seq

    def start_index(self, attr):
        attr = self.getAttributes(attr, _indexAttrMap)
        defn = ABag()
        if 'item' in attr:
            label = attr['item']
        else:
            self._syntax_error('<index> needs at least an item attribute')
        if 'name' in attr:
            name = attr['name']
        else:
            name = DEFAULT_INDEX_NAME
        format = attr.get('format', None)
        if format is not None and format not in ('123', 'I', 'i', 'ABC', 'abc'):
            raise ValueError('index tag format is %r not valid 123 I i ABC or abc' % offset)
        offset = attr.get('offset', None)
        if offset is not None:
            try:
                offset = int(offset)
            except:
                raise ValueError('index tag offset is %r not an int' % offset)
        defn.label = encode_label((label, format, offset))
        defn.name = name
        defn.kind = 'index'
        self._push('index', cbDefn=defn)
        self.handle_data('')
        self._pop('index')
    end_index = end_seq

    def start_unknown(self, attr):
        pass
    end_unknown = end_seq

    def _push(self, tag, **attr):
        frag = copy.copy(self._stack[-1])
        frag.__tag__ = tag
        _applyAttributes(frag, attr)
        self._stack.append(frag)

    def _pop(self, tag):
        frag = self._stack.pop()
        if tag == frag.__tag__:
            return frag
        raise ValueError('Parse error: saw </%s> instead of expected </%s>' % (tag, frag.__tag__))

    def getAttributes(self, attr, attrMap):
        A = {}
        for k, v in attr.items():
            if not self.caseSensitive:
                k = k.lower()
            if k in attrMap:
                j = attrMap[k]
                func = j[1]
                if func is not None:
                    v = func(self, v) if isinstance(func, _ExValidate) else func(v)
                A[j[0]] = v
            else:
                self._syntax_error('invalid attribute name %s attrMap=%r' % (k, list(sorted(attrMap.keys()))))
        return A

    def __init__(self, verbose=0, caseSensitive=0, ignoreUnknownTags=1, crashOnError=True):
        HTMLParser.__init__(self, **dict(convert_charrefs=False))
        self.verbose = verbose
        self.caseSensitive = caseSensitive
        self.ignoreUnknownTags = ignoreUnknownTags
        self._crashOnError = crashOnError

    def _iReset(self):
        self.fragList = []
        if hasattr(self, 'bFragList'):
            delattr(self, 'bFragList')

    def _reset(self, style):
        """reset the parser"""
        HTMLParser.reset(self)
        self.errors = []
        self._style = style
        self._iReset()

    def handle_data(self, data):
        """Creates an intermediate representation of string segments."""
        frag = copy.copy(self._stack[-1])
        if hasattr(frag, 'cbDefn'):
            kind = frag.cbDefn.kind
            if data:
                self._syntax_error('Only empty <%s> tag allowed' % kind)
        elif hasattr(frag, '_selfClosingTag'):
            if data != '':
                self._syntax_error('No content allowed in %s tag' % frag._selfClosingTag)
            return
        elif frag.greek:
            frag.fontName = 'symbol'
            data = _greekConvert(data)
        frag.fontName = tt2ps(frag.fontName, frag.bold, frag.italic)
        frag.text = data
        if hasattr(frag, 'isBullet'):
            delattr(frag, 'isBullet')
            self.bFragList.append(frag)
        else:
            self.fragList.append(frag)

    def handle_cdata(self, data):
        self.handle_data(data)

    def _setup_for_parse(self, style):
        self._seq = reportlab.lib.sequencer.getSequencer()
        self._reset(style)

    def _complete_parse(self):
        """Reset after parsing, to be ready for next paragraph"""
        if self._stack:
            self._syntax_error('parse ended with %d unclosed tags\n %s' % (len(self._stack), '\n '.join((x.__tag__ for x in reversed(self._stack)))))
        del self._seq
        style = self._style
        del self._style
        if len(self.errors) == 0:
            fragList = self.fragList
            bFragList = hasattr(self, 'bFragList') and self.bFragList or None
            self._iReset()
        else:
            fragList = bFragList = None
        return (style, fragList, bFragList)

    def _tt_handle(self, tt):
        """Iterate through a pre-parsed tuple tree (e.g. from pyrxp)"""
        tag = tt[0]
        try:
            start = getattr(self, 'start_' + tag)
            end = getattr(self, 'end_' + tag)
        except AttributeError:
            if not self.ignoreUnknownTags:
                raise ValueError('Invalid tag "%s"' % tag)
            start = self.start_unknown
            end = self.end_unknown
        start(tt[1] or {})
        C = tt[2]
        if C:
            M = self._tt_handlers
            for c in C:
                M[isinstance(c, (list, tuple))](c)
        end()

    def _tt_start(self, tt):
        self._tt_handlers = (self.handle_data, self._tt_handle)
        self._tt_handle(tt)

    def tt_parse(self, tt, style):
        """parse from tupletree form"""
        self._setup_for_parse(style)
        self._tt_start(tt)
        return self._complete_parse()

    def findSpanStyle(self, style):
        raise ValueError('findSpanStyle not implemented in this parser')

    def parse(self, text, style):
        """attempt replacement for parse"""
        self._setup_for_parse(style)
        text = asUnicode(text)
        if not (len(text) >= 6 and text[0] == '<' and _re_para.match(text)):
            text = u'<para>' + text + u'</para>'
        try:
            self.feed(text)
        except:
            annotateException('\nparagraph text %s caused exception' % ascii(text))
        return self._complete_parse()

    def handle_starttag(self, tag, attrs):
        """Called by HTMLParser when a tag starts"""
        if isinstance(attrs, list):
            d = {}
            for k, v in attrs:
                d[k] = v
            attrs = d
        if not self.caseSensitive:
            tag = tag.lower()
        try:
            start = getattr(self, 'start_' + tag)
        except AttributeError:
            if not self.ignoreUnknownTags:
                raise ValueError('Invalid tag "%s"' % tag)
            start = self.start_unknown
        start(attrs or {})

    def handle_endtag(self, tag):
        """Called by HTMLParser when a tag ends"""
        if not self.caseSensitive:
            tag = tag.lower()
        try:
            end = getattr(self, 'end_' + tag)
        except AttributeError:
            if not self.ignoreUnknownTags:
                raise ValueError('Invalid tag "%s"' % tag)
            end = self.end_unknown
        end()

    def handle_entityref(self, name):
        """Handles a named entity.  """
        try:
            v = known_entities[name]
        except:
            v = u'&%s;' % name
        self.handle_data(v)