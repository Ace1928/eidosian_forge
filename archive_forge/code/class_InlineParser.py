from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
class InlineParser(object):
    """INLINE PARSER

    These are methods of an InlineParser class, defined below.
    An InlineParser keeps track of a subject (a string to be
    parsed) and a position in that subject.
    """

    def __init__(self, options={}):
        self.subject = ''
        self.brackets = None
        self.pos = 0
        self.refmap = {}
        self.options = options

    def match(self, regexString):
        """
        If regexString matches at current position in the subject, advance
        position in subject and return the match; otherwise return None.
        """
        match = re.search(regexString, self.subject[self.pos:])
        if match is None:
            return None
        else:
            self.pos += match.end()
            return match.group()

    def peek(self):
        """ Returns the character at the current subject position, or None if
        there are no more characters."""
        if self.pos < len(self.subject):
            return self.subject[self.pos]
        else:
            return None

    def spnl(self):
        """ Parse zero or more space characters, including at
        most one newline."""
        self.match(reSpnl)
        return True

    def parseBackticks(self, block):
        """ Attempt to parse backticks, adding either a backtick code span or a
        literal sequence of backticks to the 'inlines' list."""
        ticks = self.match(reTicksHere)
        if ticks is None:
            return False
        after_open_ticks = self.pos
        matched = self.match(reTicks)
        while matched is not None:
            if matched == ticks:
                node = Node('code', None)
                contents = self.subject[after_open_ticks:self.pos - len(ticks)].replace('\n', ' ')
                if contents.lstrip(' ') and contents[0] == contents[-1] == ' ':
                    node.literal = contents[1:-1]
                else:
                    node.literal = contents
                block.append_child(node)
                return True
            matched = self.match(reTicks)
        self.pos = after_open_ticks
        block.append_child(text(ticks))
        return True

    def parseBackslash(self, block):
        """
        Parse a backslash-escaped special character, adding either the
        escaped character, a hard line break (if the backslash is followed
        by a newline), or a literal backslash to the block's children.
        Assumes current character is a backslash.
        """
        subj = self.subject
        self.pos += 1
        try:
            subjchar = subj[self.pos]
        except IndexError:
            subjchar = None
        if self.peek() == '\n':
            self.pos += 1
            node = Node('linebreak', None)
            block.append_child(node)
        elif subjchar and re.search(reEscapable, subjchar):
            block.append_child(text(subjchar))
            self.pos += 1
        else:
            block.append_child(text('\\'))
        return True

    def parseAutolink(self, block):
        """Attempt to parse an autolink (URL or email in pointy brackets)."""
        m = self.match(reEmailAutolink)
        if m:
            dest = m[1:-1]
            node = Node('link', None)
            node.destination = normalize_uri('mailto:' + dest)
            node.title = ''
            node.append_child(text(dest))
            block.append_child(node)
            return True
        else:
            m = self.match(reAutolink)
            if m:
                dest = m[1:-1]
                node = Node('link', None)
                node.destination = normalize_uri(dest)
                node.title = ''
                node.append_child(text(dest))
                block.append_child(node)
                return True
        return False

    def parseHtmlTag(self, block):
        """Attempt to parse a raw HTML tag."""
        m = self.match(common.reHtmlTag)
        if m is None:
            return False
        else:
            node = Node('html_inline', None)
            node.literal = m
            block.append_child(node)
            return True

    def scanDelims(self, c):
        """
        Scan a sequence of characters == c, and return information about
        the number of delimiters and whether they are positioned such that
        they can open and/or close emphasis or strong emphasis.  A utility
        function for strong/emph parsing.
        """
        numdelims = 0
        startpos = self.pos
        if c == "'" or c == '"':
            numdelims += 1
            self.pos += 1
        else:
            while self.peek() == c:
                numdelims += 1
                self.pos += 1
        if numdelims == 0:
            return None
        c_before = '\n' if startpos == 0 else self.subject[startpos - 1]
        c_after = self.peek()
        if c_after is None:
            c_after = '\n'
        after_is_whitespace = re.search(reUnicodeWhitespaceChar, c_after) or c_after == '\xa0'
        after_is_punctuation = re.search(rePunctuation, c_after)
        before_is_whitespace = re.search(reUnicodeWhitespaceChar, c_before) or c_before == '\xa0'
        before_is_punctuation = re.search(rePunctuation, c_before)
        left_flanking = not after_is_whitespace and (not after_is_punctuation or before_is_whitespace or before_is_punctuation)
        right_flanking = not before_is_whitespace and (not before_is_punctuation or after_is_whitespace or after_is_punctuation)
        if c == '_':
            can_open = left_flanking and (not right_flanking or before_is_punctuation)
            can_close = right_flanking and (not left_flanking or after_is_punctuation)
        elif c == "'" or c == '"':
            can_open = left_flanking and (not right_flanking)
            can_close = right_flanking
        else:
            can_open = left_flanking
            can_close = right_flanking
        self.pos = startpos
        return {'numdelims': numdelims, 'can_open': can_open, 'can_close': can_close}

    def handleDelim(self, cc, block):
        """Handle a delimiter marker for emphasis or a quote."""
        res = self.scanDelims(cc)
        if not res:
            return False
        numdelims = res.get('numdelims')
        startpos = self.pos
        self.pos += numdelims
        if cc == "'":
            contents = '’'
        elif cc == '"':
            contents = '“'
        else:
            contents = self.subject[startpos:self.pos]
        node = text(contents)
        block.append_child(node)
        self.delimiters = {'cc': cc, 'numdelims': numdelims, 'origdelims': numdelims, 'node': node, 'previous': self.delimiters, 'next': None, 'can_open': res.get('can_open'), 'can_close': res.get('can_close')}
        if self.delimiters['previous'] is not None:
            self.delimiters['previous']['next'] = self.delimiters
        return True

    def removeDelimiter(self, delim):
        if delim.get('previous') is not None:
            delim['previous']['next'] = delim.get('next')
        if delim.get('next') is None:
            self.delimiters = delim.get('previous')
        else:
            delim['next']['previous'] = delim.get('previous')

    @staticmethod
    def removeDelimitersBetween(bottom, top):
        if bottom.get('next') != top:
            bottom['next'] = top
            top['previous'] = bottom

    def processEmphasis(self, stack_bottom):
        openers_bottom = {'_': stack_bottom, '*': stack_bottom, "'": stack_bottom, '"': stack_bottom}
        odd_match = False
        use_delims = 0
        closer = self.delimiters
        while closer is not None and closer.get('previous') != stack_bottom:
            closer = closer.get('previous')
        while closer is not None:
            if not closer.get('can_close'):
                closer = closer.get('next')
            else:
                opener = closer.get('previous')
                opener_found = False
                closercc = closer.get('cc')
                while opener is not None and opener != stack_bottom and (opener != openers_bottom[closercc]):
                    odd_match = (closer.get('can_open') or opener.get('can_close')) and closer['origdelims'] % 3 != 0 and ((opener['origdelims'] + closer['origdelims']) % 3 == 0)
                    if opener.get('cc') == closercc and opener.get('can_open') and (not odd_match):
                        opener_found = True
                        break
                    opener = opener.get('previous')
                old_closer = closer
                if closercc == '*' or closercc == '_':
                    if not opener_found:
                        closer = closer.get('next')
                    else:
                        use_delims = 2 if closer['numdelims'] >= 2 and opener['numdelims'] >= 2 else 1
                        opener_inl = opener.get('node')
                        closer_inl = closer.get('node')
                        opener['numdelims'] -= use_delims
                        closer['numdelims'] -= use_delims
                        opener_inl.literal = opener_inl.literal[:len(opener_inl.literal) - use_delims]
                        closer_inl.literal = closer_inl.literal[:len(closer_inl.literal) - use_delims]
                        if use_delims == 1:
                            emph = Node('emph', None)
                        else:
                            emph = Node('strong', None)
                        tmp = opener_inl.nxt
                        while tmp and tmp != closer_inl:
                            nxt = tmp.nxt
                            tmp.unlink()
                            emph.append_child(tmp)
                            tmp = nxt
                        opener_inl.insert_after(emph)
                        self.removeDelimitersBetween(opener, closer)
                        if opener['numdelims'] == 0:
                            opener_inl.unlink()
                            self.removeDelimiter(opener)
                        if closer['numdelims'] == 0:
                            closer_inl.unlink()
                            tempstack = closer['next']
                            self.removeDelimiter(closer)
                            closer = tempstack
                elif closercc == "'":
                    closer['node'].literal = '’'
                    if opener_found:
                        opener['node'].literal = '‘'
                    closer = closer['next']
                elif closercc == '"':
                    closer['node'].literal = '”'
                    if opener_found:
                        opener['node'].literal = '“'
                    closer = closer['next']
                if not opener_found and (not odd_match):
                    openers_bottom[closercc] = old_closer['previous']
                    if not old_closer['can_open']:
                        self.removeDelimiter(old_closer)
        while self.delimiters is not None and self.delimiters != stack_bottom:
            self.removeDelimiter(self.delimiters)

    def parseLinkTitle(self):
        """
        Attempt to parse link title (sans quotes), returning the string
        or None if no match.
        """
        title = self.match(reLinkTitle)
        if title is None:
            return None
        else:
            return unescape_string(title[1:-1])

    def parseLinkDestination(self):
        """
        Attempt to parse link destination, returning the string or
        None if no match.
        """
        res = self.match(reLinkDestinationBraces)
        if res is None:
            if self.peek() == '<':
                return None
            savepos = self.pos
            openparens = 0
            while True:
                c = self.peek()
                if c is None:
                    break
                if c == '\\' and re.search(reEscapable, self.subject[self.pos + 1:self.pos + 2]):
                    self.pos += 1
                    if self.peek() is not None:
                        self.pos += 1
                elif c == '(':
                    self.pos += 1
                    openparens += 1
                elif c == ')':
                    if openparens < 1:
                        break
                    else:
                        self.pos += 1
                        openparens -= 1
                elif re.search(reWhitespaceChar, c):
                    break
                else:
                    self.pos += 1
            if self.pos == savepos and c != ')':
                return None
            res = self.subject[savepos:self.pos]
            return normalize_uri(unescape_string(res))
        else:
            return normalize_uri(unescape_string(res[1:-1]))

    def parseLinkLabel(self):
        """
        Attempt to parse a link label, returning number of
        characters parsed.
        """
        m = self.match(reLinkLabel)
        if m is None or len(m) > 1001:
            return 0
        else:
            return len(m)

    def parseOpenBracket(self, block):
        """
        Add open bracket to delimiter stack and add a text node to
        block's children.
        """
        startpos = self.pos
        self.pos += 1
        node = text('[')
        block.append_child(node)
        self.addBracket(node, startpos, False)
        return True

    def parseBang(self, block):
        """
        If next character is [, and ! delimiter to delimiter stack and
        add a text node to block's children. Otherwise just add a text
        node.
        """
        startpos = self.pos
        self.pos += 1
        if self.peek() == '[':
            self.pos += 1
            node = text('![')
            block.append_child(node)
            self.addBracket(node, startpos + 1, True)
        else:
            block.append_child(text('!'))
        return True

    def parseCloseBracket(self, block):
        """
        Try to match close bracket against an opening in the delimiter
        stack. Add either a link or image, or a plain [ character,
        to block's children. If there is a matching delimiter,
        remove it from the delimiter stack.
        """
        title = None
        matched = False
        self.pos += 1
        startpos = self.pos
        opener = self.brackets
        if opener is None:
            block.append_child(text(']'))
            return True
        if not opener.get('active'):
            block.append_child(text(']'))
            self.removeBracket()
            return True
        is_image = opener.get('image')
        savepos = self.pos
        if self.peek() == '(':
            self.pos += 1
            self.spnl()
            dest = self.parseLinkDestination()
            if dest is not None and self.spnl():
                if re.search(reWhitespaceChar, self.subject[self.pos - 1]):
                    title = self.parseLinkTitle()
                if self.spnl() and self.peek() == ')':
                    self.pos += 1
                    matched = True
            else:
                self.pos = savepos
        if not matched:
            beforelabel = self.pos
            n = self.parseLinkLabel()
            if n > 2:
                reflabel = self.subject[beforelabel:beforelabel + n]
            elif not opener.get('bracket_after'):
                reflabel = self.subject[opener.get('index'):startpos]
            if n == 0:
                self.pos = savepos
            if reflabel:
                link = self.refmap.get(normalize_reference(reflabel))
                if link:
                    dest = link['destination']
                    title = link['title']
                    matched = True
        if matched:
            node = Node('image' if is_image else 'link', None)
            node.destination = dest
            node.title = title or ''
            tmp = opener.get('node').nxt
            while tmp:
                nxt = tmp.nxt
                tmp.unlink()
                node.append_child(tmp)
                tmp = nxt
            block.append_child(node)
            self.processEmphasis(opener.get('previousDelimiter'))
            self.removeBracket()
            opener.get('node').unlink()
            if not is_image:
                opener = self.brackets
                while opener is not None:
                    if not opener.get('image'):
                        opener['active'] = False
                    opener = opener.get('previous')
            return True
        else:
            self.removeBracket()
            self.pos = startpos
            block.append_child(text(']'))
            return True

    def addBracket(self, node, index, image):
        if self.brackets is not None:
            self.brackets['bracketAfter'] = True
        self.brackets = {'node': node, 'previous': self.brackets, 'previousDelimiter': self.delimiters, 'index': index, 'image': image, 'active': True}

    def removeBracket(self):
        self.brackets = self.brackets.get('previous')

    def parseEntity(self, block):
        """Attempt to parse an entity."""
        m = self.match(reEntityHere)
        if m:
            block.append_child(text(HTMLunescape(m)))
            return True
        else:
            return False

    def parseString(self, block):
        """
        Parse a run of ordinary characters, or a single character with
        a special meaning in markdown, as a plain string.
        """
        m = self.match(reMain)
        if m:
            if self.options.get('smart'):
                s = re.sub(reEllipses, '…', m)
                s = re.sub(reDash, lambda x: smart_dashes(x.group()), s)
                block.append_child(text(s))
            else:
                block.append_child(text(m))
            return True
        else:
            return False

    def parseNewline(self, block):
        """
        Parse a newline.  If it was preceded by two spaces, return a hard
        line break; otherwise a soft line break.
        """
        self.pos += 1
        lastc = block.last_child
        if lastc and lastc.t == 'text' and (lastc.literal[-1] == ' '):
            linebreak = len(lastc.literal) >= 2 and lastc.literal[-2] == ' '
            lastc.literal = re.sub(reFinalSpace, '', lastc.literal)
            if linebreak:
                node = Node('linebreak', None)
            else:
                node = Node('softbreak', None)
            block.append_child(node)
        else:
            block.append_child(Node('softbreak', None))
        self.match(reInitialSpace)
        return True

    def parseReference(self, s, refmap):
        """Attempt to parse a link reference, modifying refmap."""
        self.subject = s
        self.pos = 0
        startpos = self.pos
        match_chars = self.parseLinkLabel()
        if match_chars == 0 or match_chars == 2:
            return 0
        else:
            rawlabel = self.subject[:match_chars]
        if self.peek() == ':':
            self.pos += 1
        else:
            self.pos = startpos
            return 0
        self.spnl()
        dest = self.parseLinkDestination()
        if dest is None:
            self.pos = startpos
            return 0
        beforetitle = self.pos
        self.spnl()
        title = None
        if self.pos != beforetitle:
            title = self.parseLinkTitle()
        if title is None:
            title = ''
            self.pos = beforetitle
        at_line_end = True
        if self.match(reSpaceAtEndOfLine) is None:
            if title == '':
                at_line_end = False
            else:
                title == ''
                self.pos = beforetitle
                at_line_end = self.match(reSpaceAtEndOfLine) is not None
        if not at_line_end:
            self.pos = startpos
            return 0
        normlabel = normalize_reference(rawlabel)
        if normlabel == '':
            self.pos = startpos
            return 0
        if not refmap.get(normlabel):
            refmap[normlabel] = {'destination': dest, 'title': title}
        return self.pos - startpos

    def parseInline(self, block):
        """
        Parse the next inline element in subject, advancing subject
        position.

        On success, add the result to block's children and return True.
        On failure, return False.
        """
        res = False
        c = self.peek()
        if c is None:
            return False
        if c == '\n':
            res = self.parseNewline(block)
        elif c == '\\':
            res = self.parseBackslash(block)
        elif c == '`':
            res = self.parseBackticks(block)
        elif c == '*' or c == '_':
            res = self.handleDelim(c, block)
        elif c == "'" or c == '"':
            res = self.options.get('smart') and self.handleDelim(c, block)
        elif c == '[':
            res = self.parseOpenBracket(block)
        elif c == '!':
            res = self.parseBang(block)
        elif c == ']':
            res = self.parseCloseBracket(block)
        elif c == '<':
            res = self.parseAutolink(block) or self.parseHtmlTag(block)
        elif c == '&':
            res = self.parseEntity(block)
        else:
            res = self.parseString(block)
        if not res:
            self.pos += 1
            block.append_child(text(c))
        return True

    def parseInlines(self, block):
        """
        Parse string content in block into inline children,
        using refmap to resolve references.
        """
        self.subject = block.string_content.strip()
        self.pos = 0
        self.delimiters = None
        self.brackets = None
        while self.parseInline(block):
            pass
        block.string_content = None
        self.processEmphasis(None)
    parse = parseInlines