import re
from kivy.properties import dpi2px
from kivy.parser import parse_color
from kivy.logger import Logger
from kivy.core.text import Label, LabelBase
from kivy.core.text.text_layout import layout_text, LayoutWord, LayoutLine
from copy import copy
from functools import partial
class MarkupLabel(MarkupLabelBase):
    """Markup text label.

    See module documentation for more information.
    """

    def __init__(self, *largs, **kwargs):
        self._style_stack = {}
        self._refs = {}
        self._anchors = {}
        super(MarkupLabel, self).__init__(*largs, **kwargs)
        self._internal_size = (0, 0)
        self._cached_lines = []

    @property
    def refs(self):
        """Get the bounding box of all the ``[ref=...]``::

            { 'refA': ((x1, y1, x2, y2), (x1, y1, x2, y2)), ... }
        """
        return self._refs

    @property
    def anchors(self):
        """Get the position of all the ``[anchor=...]``::

            { 'anchorA': (x, y), 'anchorB': (x, y), ... }
        """
        return self._anchors

    @property
    def markup(self):
        """Return the text with all the markup split::

            >>> MarkupLabel('[b]Hello world[/b]').markup
            >>> ('[b]', 'Hello world', '[/b]')

        """
        s = re.split('(\\[.*?\\])', self.label)
        s = [x for x in s if x != '']
        return s

    def _push_style(self, k):
        if k not in self._style_stack:
            self._style_stack[k] = []
        self._style_stack[k].append(self.options[k])

    def _pop_style(self, k):
        if k not in self._style_stack or len(self._style_stack[k]) == 0:
            Logger.warning('Label: pop style stack without push')
            return
        v = self._style_stack[k].pop()
        self.options[k] = v

    def render(self, real=False):
        options = copy(self.options)
        if not real:
            ret = self._pre_render()
        else:
            ret = self._render_real()
        self.options = options
        return ret

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

    def render_lines(self, lines, options, render_text, y, size):
        padding_left = options['padding'][0]
        padding_right = options['padding'][2]
        w = size[0]
        halign = options['halign']
        refs = self._refs
        anchors = self._anchors
        base_dir = options['base_direction'] or self._resolved_base_dir
        auto_halign_r = halign == 'auto' and base_dir and ('rtl' in base_dir)
        for layout_line in lines:
            lw, lh = (layout_line.w, layout_line.h)
            x = padding_left
            if halign == 'center':
                x = min(int(w - lw), max(int(padding_left), int((w - lw + padding_left - padding_right) / 2.0)))
            elif halign == 'right' or auto_halign_r:
                x = max(0, int(w - lw - padding_right))
            layout_line.x = x
            layout_line.y = y
            psp = pph = 0
            for word in layout_line.words:
                options = self.options = word.options
                wh = options['line_height'] * word.lh
                if options['script'] == 'superscript':
                    script_pos = max(0, psp if psp and wh < pph else self.get_descent())
                    psp = script_pos
                    pph = wh
                elif options['script'] == 'subscript':
                    script_pos = min(lh - wh, psp + pph - wh if pph and wh < pph else lh - wh)
                    pph = wh
                    psp = script_pos
                else:
                    script_pos = (lh - wh) / 1.25
                    psp = pph = 0
                if len(word.text):
                    render_text(word.text, x, y + script_pos)
                ref = options['_ref']
                if ref is not None:
                    if ref not in refs:
                        refs[ref] = []
                    refs[ref].append((x, y, x + word.lw, y + wh))
                anchor = options['_anchor']
                if anchor is not None:
                    if anchor not in anchors:
                        anchors[anchor] = (x, y)
                x += word.lw
            y += lh
        return y

    def shorten_post(self, lines, w, h, margin=2):
        """ Shortens the text to a single line according to the label options.

        This function operates on a text that has already been laid out because
        for markup, parts of text can have different size and options.

        If :attr:`text_size` [0] is None, the lines are returned unchanged.
        Otherwise, the lines are converted to a single line fitting within the
        constrained width, :attr:`text_size` [0].

        :params:

            `lines`: list of `LayoutLine` instances describing the text.
            `w`: int, the width of the text in lines, including padding.
            `h`: int, the height of the text in lines, including padding.
            `margin` int, the additional space left on the sides. This is in
            addition to :attr:`padding_x`.

        :returns:
            3-tuple of (xw, h, lines), where w, and h is similar to the input
            and contains the resulting width / height of the text, including
            padding. lines, is a list containing a single `LayoutLine`, which
            contains the words for the line.
        """

        def n(line, c):
            """ A function similar to text.find, except it's an iterator that
            returns successive occurrences of string c in list line. line is
            not a string, but a list of LayoutWord instances that we walk
            from left to right returning the indices of c in the words as we
            encounter them. Note that the options can be different among the
            words.

            :returns:
                3-tuple: the index of the word in line, the index of the
                occurrence in word, and the extents (width) of the combined
                words until this occurrence, not including the occurrence char.
                If no more are found it returns (-1, -1, total_w) where total_w
                is the full width of all the words.
            """
            total_w = 0
            for w in range(len(line)):
                word = line[w]
                if not word.lw:
                    continue
                f = partial(word.text.find, c)
                i = f()
                while i != -1:
                    self.options = word.options
                    yield (w, i, total_w + self.get_extents(word.text[:i])[0])
                    i = f(i + 1)
                self.options = word.options
                total_w += self.get_extents(word.text)[0]
            yield (-1, -1, total_w)

        def p(line, c):
            """ Similar to the `n` function, except it returns occurrences of c
            from right to left in the list, line, similar to rfind.
            """
            total_w = 0
            offset = 0 if len(c) else 1
            for w in range(len(line) - 1, -1, -1):
                word = line[w]
                if not word.lw:
                    continue
                f = partial(word.text.rfind, c)
                i = f()
                while i != -1:
                    self.options = word.options
                    yield (w, i, total_w + self.get_extents(word.text[i + 1:])[0])
                    if i:
                        i = f(0, i - offset)
                    else:
                        if not c:
                            self.options = word.options
                            yield (w, -1, total_w + self.get_extents(word.text)[0])
                        break
                self.options = word.options
                total_w += self.get_extents(word.text)[0]
            yield (-1, -1, total_w)

        def n_restricted(line, uw, c):
            """ Similar to the function `n`, except it only returns the first
            occurrence and it's not an iterator. Furthermore, if the first
            occurrence doesn't fit within width uw, it returns the index of
            whatever amount of text will still fit in uw.

            :returns:
                similar to the function `n`, except it's a 4-tuple, with the
                last element a boolean, indicating if we had to clip the text
                to fit in uw (True) or if the whole text until the first
                occurrence fitted in uw (False).
            """
            total_w = 0
            if not len(line):
                return (0, 0, 0)
            for w in range(len(line)):
                word = line[w]
                f = partial(word.text.find, c)
                self.options = word.options
                extents = self.get_cached_extents()
                i = f()
                if i != -1:
                    ww = extents(word.text[:i])[0]
                if i != -1 and total_w + ww <= uw:
                    return (w, i, total_w + ww, False)
                elif i == -1:
                    ww = extents(word.text)[0]
                    if total_w + ww <= uw:
                        total_w += ww
                        continue
                    i = len(word.text)
                e = 0
                while e != i and total_w + extents(word.text[:e])[0] <= uw:
                    e += 1
                e = max(0, e - 1)
                return (w, e, total_w + extents(word.text[:e])[0], True)
            return (-1, -1, total_w, False)

        def p_restricted(line, uw, c):
            """ Similar to `n_restricted`, except it returns the first
            occurrence starting from the right, like `p`.
            """
            total_w = 0
            if not len(line):
                return (0, 0, 0)
            for w in range(len(line) - 1, -1, -1):
                word = line[w]
                f = partial(word.text.rfind, c)
                self.options = word.options
                extents = self.get_cached_extents()
                i = f()
                if i != -1:
                    ww = extents(word.text[i + 1:])[0]
                if i != -1 and total_w + ww <= uw:
                    return (w, i, total_w + ww, False)
                elif i == -1:
                    ww = extents(word.text)[0]
                    if total_w + ww <= uw:
                        total_w += ww
                        continue
                s = len(word.text) - 1
                while s >= 0 and total_w + extents(word.text[s:])[0] <= uw:
                    s -= 1
                return (w, s, total_w + extents(word.text[s + 1:])[0], True)
            return (-1, -1, total_w, False)
        textwidth = self.get_cached_extents()
        uw = self.text_size[0]
        if uw is None:
            return (w, h, lines)
        old_opts = copy(self.options)
        uw = max(0, int(uw - old_opts['padding'][0] - old_opts['padding'][2] - margin))
        chr = type(self.text)
        ssize = textwidth(' ')
        c = old_opts['split_str']
        line_height = old_opts['line_height']
        xpad, ypad = (old_opts['padding'][0] + old_opts['padding'][2], old_opts['padding'][1] + old_opts['padding'][3])
        dir = old_opts['shorten_from'][0]
        line = []
        last_w = 0
        for l in range(len(lines)):
            this_line = lines[l]
            if last_w and this_line.w and (not this_line.line_wrap):
                line.append(LayoutWord(old_opts, ssize[0], ssize[1], chr(' ')))
            last_w = this_line.w or last_w
            for word in this_line.words:
                if word.lw:
                    line.append(word)
        lw = sum([word.lw for word in line])
        if lw <= uw:
            lh = max([word.lh for word in line] + [0]) * line_height
            self.is_shortened = False
            return (lw + xpad, lh + ypad, [LayoutLine(0, 0, lw, lh, 1, 0, line)])
        elps_opts = copy(old_opts)
        if 'ellipsis_options' in old_opts:
            elps_opts.update(old_opts['ellipsis_options'])
        self.options = elps_opts
        elps_s = textwidth('...')
        if elps_s[0] > uw:
            self.is_shortened = True
            s = textwidth('..')
            if s[0] <= uw:
                return (s[0] + xpad, s[1] * line_height + ypad, [LayoutLine(0, 0, s[0], s[1], 1, 0, [LayoutWord(old_opts, s[0], s[1], '..')])])
            else:
                s = textwidth('.')
                return (s[0] + xpad, s[1] * line_height + ypad, [LayoutLine(0, 0, s[0], s[1], 1, 0, [LayoutWord(old_opts, s[0], s[1], '.')])])
        elps = LayoutWord(elps_opts, elps_s[0], elps_s[1], '...')
        uw -= elps_s[0]
        self.options = old_opts
        w1, e1, l1, clipped1 = n_restricted(line, uw, c)
        w2, s2, l2, clipped2 = p_restricted(line, uw, c)
        if dir != 'l':
            line1 = None
            if clipped1 or clipped2 or l1 + l2 > uw:
                if len(c):
                    self.options = old_opts
                    old_opts['split_str'] = ''
                    res = self.shorten_post(lines, w, h, margin)
                    self.options['split_str'] = c
                    self.is_shortened = True
                    return res
                line1 = line[:w1]
                last_word = line[w1]
                last_text = last_word.text[:e1]
                self.options = last_word.options
                s = self.get_extents(last_text)
                line1.append(LayoutWord(last_word.options, s[0], s[1], last_text))
            elif (w1, e1) == (-1, -1):
                line1 = line
            if line1:
                line1.append(elps)
                lw = sum([word.lw for word in line1])
                lh = max([word.lh for word in line1]) * line_height
                self.options = old_opts
                self.is_shortened = True
                return (lw + xpad, lh + ypad, [LayoutLine(0, 0, lw, lh, 1, 0, line1)])
            if (w1, e1) != (w2, s2):
                if dir == 'r':
                    f = n(line, c)
                    assert next(f)[:-1] == (w1, e1)
                    ww1, ee1, l1 = next(f)
                    while l2 + l1 <= uw:
                        w1, e1 = (ww1, ee1)
                        ww1, ee1, l1 = next(f)
                        if (w1, e1) == (w2, s2):
                            break
                else:
                    f = n(line, c)
                    f_inv = p(line, c)
                    assert next(f)[:-1] == (w1, e1)
                    assert next(f_inv)[:-1] == (w2, s2)
                    while True:
                        if l1 <= l2:
                            ww1, ee1, l1 = next(f)
                            if l2 + l1 > uw:
                                break
                            w1, e1 = (ww1, ee1)
                            if (w1, e1) == (w2, s2):
                                break
                        else:
                            ww2, ss2, l2 = next(f_inv)
                            if l2 + l1 > uw:
                                break
                            w2, s2 = (ww2, ss2)
                            if (w1, e1) == (w2, s2):
                                break
        else:
            line1 = [elps]
            if clipped1 or clipped2 or l1 + l2 > uw:
                if len(c):
                    self.options = old_opts
                    old_opts['split_str'] = ''
                    res = self.shorten_post(lines, w, h, margin)
                    self.options['split_str'] = c
                    self.is_shortened = True
                    return res
                first_word = line[w2]
                first_text = first_word.text[s2 + 1:]
                self.options = first_word.options
                s = self.get_extents(first_text)
                line1.append(LayoutWord(first_word.options, s[0], s[1], first_text))
                line1.extend(line[w2 + 1:])
            elif (w1, e1) == (-1, -1):
                line1 = line
            if len(line1) != 1:
                lw = sum([word.lw for word in line1])
                lh = max([word.lh for word in line1]) * line_height
                self.options = old_opts
                self.is_shortened = True
                return (lw + xpad, lh + ypad, [LayoutLine(0, 0, lw, lh, 1, 0, line1)])
            if (w1, e1) != (w2, s2):
                f_inv = p(line, c)
                assert next(f_inv)[:-1] == (w2, s2)
                ww2, ss2, l2 = next(f_inv)
                while l2 + l1 <= uw:
                    w2, s2 = (ww2, ss2)
                    ww2, ss2, l2 = next(f_inv)
                    if (w1, e1) == (w2, s2):
                        break
        line1 = line[:w1]
        last_word = line[w1]
        last_text = last_word.text[:e1]
        self.options = last_word.options
        s = self.get_extents(last_text)
        if len(last_text):
            line1.append(LayoutWord(last_word.options, s[0], s[1], last_text))
        line1.append(elps)
        first_word = line[w2]
        first_text = first_word.text[s2 + 1:]
        self.options = first_word.options
        s = self.get_extents(first_text)
        if len(first_text):
            line1.append(LayoutWord(first_word.options, s[0], s[1], first_text))
        line1.extend(line[w2 + 1:])
        lw = sum([word.lw for word in line1])
        lh = max([word.lh for word in line1]) * line_height
        self.options = old_opts
        if uw < lw:
            self.is_shortened = True
        return (lw + xpad, lh + ypad, [LayoutLine(0, 0, lw, lh, 1, 0, line1)])