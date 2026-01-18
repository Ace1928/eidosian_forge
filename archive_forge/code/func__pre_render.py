import re
from kivy.properties import dpi2px
from kivy.parser import parse_color
from kivy.logger import Logger
from kivy.core.text import Label, LabelBase
from kivy.core.text.text_layout import layout_text, LayoutWord, LayoutLine
from copy import copy
from functools import partial
def _pre_render(self):
    self._cached_lines = lines = []
    self._refs = {}
    self._anchors = {}
    clipped = False
    w = h = 0
    uw, uh = self.text_size
    spush = self._push_style
    spop = self._pop_style
    options = self.options
    options['_ref'] = None
    options['_anchor'] = None
    options['script'] = 'normal'
    shorten = options['shorten']
    uw_temp = None if shorten else uw
    xpad = options['padding'][0] + options['padding'][2]
    uhh = None if uh is not None and options['valign'] != 'top' or options['shorten'] else uh
    options['strip'] = options['strip'] or options['halign'] == 'justify'
    find_base_dir = Label.find_base_direction
    base_dir = options['base_direction']
    self._resolved_base_dir = None
    for item in self.markup:
        if item == '[b]':
            spush('bold')
            options['bold'] = True
            self.resolve_font_name()
        elif item == '[/b]':
            spop('bold')
            self.resolve_font_name()
        elif item == '[i]':
            spush('italic')
            options['italic'] = True
            self.resolve_font_name()
        elif item == '[/i]':
            spop('italic')
            self.resolve_font_name()
        elif item == '[u]':
            spush('underline')
            options['underline'] = True
            self.resolve_font_name()
        elif item == '[/u]':
            spop('underline')
            self.resolve_font_name()
        elif item == '[s]':
            spush('strikethrough')
            options['strikethrough'] = True
            self.resolve_font_name()
        elif item == '[/s]':
            spop('strikethrough')
            self.resolve_font_name()
        elif item[:6] == '[size=':
            item = item[6:-1]
            try:
                if item[-2:] in ('px', 'pt', 'in', 'cm', 'mm', 'dp', 'sp'):
                    size = dpi2px(item[:-2], item[-2:])
                else:
                    size = int(item)
            except ValueError:
                raise
                size = options['font_size']
            spush('font_size')
            options['font_size'] = size
        elif item == '[/size]':
            spop('font_size')
        elif item[:7] == '[color=':
            color = parse_color(item[7:-1])
            spush('color')
            options['color'] = color
        elif item == '[/color]':
            spop('color')
        elif item[:6] == '[font=':
            fontname = item[6:-1]
            spush('font_name')
            options['font_name'] = fontname
            self.resolve_font_name()
        elif item == '[/font]':
            spop('font_name')
            self.resolve_font_name()
        elif item[:13] == '[font_family=':
            spush('font_family')
            options['font_family'] = item[13:-1]
        elif item == '[/font_family]':
            spop('font_family')
        elif item[:14] == '[font_context=':
            fctx = item[14:-1]
            if not fctx or fctx.lower() == 'none':
                fctx = None
            spush('font_context')
            options['font_context'] = fctx
        elif item == '[/font_context]':
            spop('font_context')
        elif item[:15] == '[font_features=':
            spush('font_features')
            options['font_features'] = item[15:-1]
        elif item == '[/font_features]':
            spop('font_features')
        elif item[:15] == '[text_language=':
            lang = item[15:-1]
            if not lang or lang.lower() == 'none':
                lang = None
            spush('text_language')
            options['text_language'] = lang
        elif item == '[/text_language]':
            spop('text_language')
        elif item[:5] == '[sub]':
            spush('font_size')
            spush('script')
            options['font_size'] = options['font_size'] * 0.5
            options['script'] = 'subscript'
        elif item == '[/sub]':
            spop('font_size')
            spop('script')
        elif item[:5] == '[sup]':
            spush('font_size')
            spush('script')
            options['font_size'] = options['font_size'] * 0.5
            options['script'] = 'superscript'
        elif item == '[/sup]':
            spop('font_size')
            spop('script')
        elif item[:5] == '[ref=':
            ref = item[5:-1]
            spush('_ref')
            options['_ref'] = ref
        elif item == '[/ref]':
            spop('_ref')
        elif not clipped and item[:8] == '[anchor=':
            options['_anchor'] = item[8:-1]
        elif not clipped:
            item = item.replace('&bl;', '[').replace('&br;', ']').replace('&amp;', '&')
            if not base_dir:
                base_dir = self._resolved_base_dir = find_base_dir(item)
            opts = copy(options)
            extents = self.get_cached_extents()
            opts['space_width'] = extents(' ')[0]
            w, h, clipped = layout_text(item, lines, (w, h), (uw_temp, uhh), opts, extents, append_down=True, complete=False)
    if len(lines):
        old_opts = self.options
        self.options = copy(opts)
        w, h, clipped = layout_text('', lines, (w, h), (uw_temp, uhh), self.options, self.get_cached_extents(), append_down=True, complete=True)
        self.options = old_opts
    self.is_shortened = False
    if shorten:
        options['_ref'] = None
        options['_anchor'] = None
        w, h, lines = self.shorten_post(lines, w, h)
        self._cached_lines = lines
    elif uh != uhh and h > uh and (len(lines) > 1):
        if options['valign'] == 'bottom':
            i = 0
            while i < len(lines) - 1 and h > uh:
                h -= lines[i].h
                i += 1
            del lines[:i]
        else:
            i = 0
            top = int(h / 2.0 + uh / 2.0)
            while i < len(lines) - 1 and h > top:
                h -= lines[i].h
                i += 1
            del lines[:i]
            i = len(lines) - 1
            while i and h > uh:
                h -= lines[i].h
                i -= 1
            del lines[i + 1:]
    if options['halign'] == 'justify' and uw is not None:
        split = partial(re.split, re.compile('( +)'))
        uww = uw - xpad
        chr = type(self.text)
        space = chr(' ')
        empty = chr('')
        for i in range(len(lines)):
            line = lines[i]
            words = line.words
            if not line.w or int(uww - line.w) <= 0 or (not len(words)) or line.is_last_line:
                continue
            done = False
            parts = [None] * len(words)
            idxs = [None] * len(words)
            for w in range(len(words)):
                word = words[w]
                sw = word.options['space_width']
                p = parts[w] = split(word.text)
                idxs[w] = [v for v in range(len(p)) if p[v].startswith(' ')]
                for k in idxs[w]:
                    if line.w + sw > uww:
                        done = True
                        break
                    line.w += sw
                    word.lw += sw
                    p[k] += space
                if done:
                    break
            if not any(idxs):
                continue
            while not done:
                for w in range(len(words)):
                    if not idxs[w]:
                        continue
                    word = words[w]
                    sw = word.options['space_width']
                    p = parts[w]
                    for k in idxs[w]:
                        if line.w + sw > uww:
                            done = True
                            break
                        line.w += sw
                        word.lw += sw
                        p[k] += space
                    if done:
                        break
            diff = int(uww - line.w)
            if diff > 0:
                for w in range(len(words) - 1, -1, -1):
                    if not idxs[w]:
                        continue
                    break
                old_opts = self.options
                self.options = word.options
                word = words[w]
                l_text = empty.join(parts[w][:idxs[w][-1]])
                r_text = empty.join(parts[w][idxs[w][-1]:])
                left = LayoutWord(word.options, self.get_extents(l_text)[0], word.lh, l_text)
                right = LayoutWord(word.options, self.get_extents(r_text)[0], word.lh, r_text)
                left.lw = max(left.lw, word.lw + diff - right.lw)
                self.options = old_opts
                for k in range(len(words)):
                    if idxs[k]:
                        words[k].text = empty.join(parts[k])
                words[w] = right
                words.insert(w, left)
            else:
                for k in range(len(words)):
                    if idxs[k]:
                        words[k].text = empty.join(parts[k])
            line.w = uww
            w = max(w, uww)
    self._internal_size = (w, h)
    if uw:
        w = uw
    if uh:
        h = uh
    if h > 1 and w < 2:
        w = 2
    if w < 1:
        w = 1
    if h < 1:
        h = 1
    return (int(w), int(h))