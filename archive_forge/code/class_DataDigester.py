from __future__ import print_function
import re
import hashlib
class DataDigester(object):
    """The major workhouse class."""
    __slots__ = ['value', 'digest']
    min_line_length = 8
    atomic_num_lines = 4
    email_ptrn = re.compile('\\S+@\\S+')
    url_ptrn = re.compile('[a-z]+:\\S+', re.IGNORECASE)
    longstr_ptrn = re.compile('\\S{10,}')
    ws_ptrn = re.compile('\\s')
    unwanted_txt_repl = ''

    def __init__(self, msg, spec=None):
        if spec is None:
            spec = digest_spec
        self.value = None
        self.digest = HASH()
        lines = []
        for payload in self.digest_payloads(msg):
            for line in payload.splitlines():
                norm = self.normalize(line)
                if self.should_handle_line(norm):
                    try:
                        lines.append(norm.encode('utf8', 'ignore'))
                    except UnicodeError:
                        continue
        if len(lines) <= self.atomic_num_lines:
            self.handle_atomic(lines)
        else:
            self.handle_pieced(lines, spec)
        self.value = self.digest.hexdigest()
        assert len(self.value) == HASH_SIZE

    def handle_atomic(self, lines):
        """We digest everything."""
        for line in lines:
            self.handle_line(line)

    def handle_pieced(self, lines, spec):
        """Digest stuff according to the spec."""
        for offset, length in spec:
            for i in xrange(length):
                try:
                    line = lines[int(offset * len(lines) // 100) + i]
                except IndexError:
                    pass
                else:
                    self.handle_line(line)

    def handle_line(self, line):
        self.digest.update(line.rstrip())

    @classmethod
    def normalize(cls, s):
        repl = cls.unwanted_txt_repl
        s = cls.longstr_ptrn.sub(repl, s)
        s = cls.email_ptrn.sub(repl, s)
        s = cls.url_ptrn.sub(repl, s)
        return cls.ws_ptrn.sub('', s).strip()

    @staticmethod
    def normalize_html_part(s):
        data = []
        stripper = HTMLStripper(data)
        try:
            stripper.feed(s)
        except (UnicodeDecodeError, HTMLParser.HTMLParseError):
            pass
        return ' '.join(data)

    @classmethod
    def should_handle_line(cls, s):
        return len(s) and cls.min_line_length <= len(s)

    @classmethod
    def digest_payloads(cls, msg):
        for part in msg.walk():
            if part.get_content_maintype() == 'text':
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset()
                errors = 'ignore'
                if not charset:
                    charset = 'ascii'
                elif charset.lower().replace('_', '-') in ('quopri-codec', 'quopri', 'quoted-printable', 'quotedprintable'):
                    errors = 'strict'
                try:
                    payload = payload.decode(charset, errors)
                except (LookupError, UnicodeError, AssertionError):
                    try:
                        payload = payload.decode('ascii', 'ignore')
                    except UnicodeError:
                        continue
                if part.get_content_subtype() == 'html':
                    yield cls.normalize_html_part(payload)
                else:
                    yield payload
            elif part.is_multipart():
                pass
            else:
                yield part.get_payload()