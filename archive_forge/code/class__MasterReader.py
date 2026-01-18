from __future__ import generators
import sys
import re
import os
from io import BytesIO
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.ttl
import dns.grange
from ._compat import string_types, text_type, PY3
class _MasterReader(object):
    """Read a DNS master file

    @ivar tok: The tokenizer
    @type tok: dns.tokenizer.Tokenizer object
    @ivar last_ttl: The last seen explicit TTL for an RR
    @type last_ttl: int
    @ivar last_ttl_known: Has last TTL been detected
    @type last_ttl_known: bool
    @ivar default_ttl: The default TTL from a $TTL directive or SOA RR
    @type default_ttl: int
    @ivar default_ttl_known: Has default TTL been detected
    @type default_ttl_known: bool
    @ivar last_name: The last name read
    @type last_name: dns.name.Name object
    @ivar current_origin: The current origin
    @type current_origin: dns.name.Name object
    @ivar relativize: should names in the zone be relativized?
    @type relativize: bool
    @ivar zone: the zone
    @type zone: dns.zone.Zone object
    @ivar saved_state: saved reader state (used when processing $INCLUDE)
    @type saved_state: list of (tokenizer, current_origin, last_name, file,
    last_ttl, last_ttl_known, default_ttl, default_ttl_known) tuples.
    @ivar current_file: the file object of the $INCLUDed file being parsed
    (None if no $INCLUDE is active).
    @ivar allow_include: is $INCLUDE allowed?
    @type allow_include: bool
    @ivar check_origin: should sanity checks of the origin node be done?
    The default is True.
    @type check_origin: bool
    """

    def __init__(self, tok, origin, rdclass, relativize, zone_factory=Zone, allow_include=False, check_origin=True):
        if isinstance(origin, string_types):
            origin = dns.name.from_text(origin)
        self.tok = tok
        self.current_origin = origin
        self.relativize = relativize
        self.last_ttl = 0
        self.last_ttl_known = False
        self.default_ttl = 0
        self.default_ttl_known = False
        self.last_name = self.current_origin
        self.zone = zone_factory(origin, rdclass, relativize=relativize)
        self.saved_state = []
        self.current_file = None
        self.allow_include = allow_include
        self.check_origin = check_origin

    def _eat_line(self):
        while 1:
            token = self.tok.get()
            if token.is_eol_or_eof():
                break

    def _rr_line(self):
        """Process one line from a DNS master file."""
        if self.current_origin is None:
            raise UnknownOrigin
        token = self.tok.get(want_leading=True)
        if not token.is_whitespace():
            self.last_name = dns.name.from_text(token.value, self.current_origin)
        else:
            token = self.tok.get()
            if token.is_eol_or_eof():
                return
            self.tok.unget(token)
        name = self.last_name
        if not name.is_subdomain(self.zone.origin):
            self._eat_line()
            return
        if self.relativize:
            name = name.relativize(self.zone.origin)
        token = self.tok.get()
        if not token.is_identifier():
            raise dns.exception.SyntaxError
        try:
            ttl = dns.ttl.from_text(token.value)
            self.last_ttl = ttl
            self.last_ttl_known = True
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.ttl.BadTTL:
            if not (self.last_ttl_known or self.default_ttl_known):
                raise dns.exception.SyntaxError('Missing default TTL value')
            if self.default_ttl_known:
                ttl = self.default_ttl
            else:
                ttl = self.last_ttl
        try:
            rdclass = dns.rdataclass.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            rdclass = self.zone.rdclass
        if rdclass != self.zone.rdclass:
            raise dns.exception.SyntaxError("RR class is not zone's class")
        try:
            rdtype = dns.rdatatype.from_text(token.value)
        except:
            raise dns.exception.SyntaxError("unknown rdatatype '%s'" % token.value)
        n = self.zone.nodes.get(name)
        if n is None:
            n = self.zone.node_factory()
            self.zone.nodes[name] = n
        try:
            rd = dns.rdata.from_text(rdclass, rdtype, self.tok, self.current_origin, False)
        except dns.exception.SyntaxError:
            ty, va = sys.exc_info()[:2]
            raise va
        except:
            ty, va = sys.exc_info()[:2]
            raise dns.exception.SyntaxError('caught exception {}: {}'.format(str(ty), str(va)))
        if not self.default_ttl_known and isinstance(rd, dns.rdtypes.ANY.SOA.SOA):
            self.default_ttl = rd.minimum
            self.default_ttl_known = True
        rd.choose_relativity(self.zone.origin, self.relativize)
        covers = rd.covers()
        rds = n.find_rdataset(rdclass, rdtype, covers, True)
        rds.add(rd, ttl)

    def _parse_modify(self, side):
        is_generate1 = re.compile('^.*\\$({(\\+|-?)(\\d+),(\\d+),(.)}).*$')
        is_generate2 = re.compile('^.*\\$({(\\+|-?)(\\d+)}).*$')
        is_generate3 = re.compile('^.*\\$({(\\+|-?)(\\d+),(\\d+)}).*$')
        g1 = is_generate1.match(side)
        if g1:
            mod, sign, offset, width, base = g1.groups()
            if sign == '':
                sign = '+'
        g2 = is_generate2.match(side)
        if g2:
            mod, sign, offset = g2.groups()
            if sign == '':
                sign = '+'
            width = 0
            base = 'd'
        g3 = is_generate3.match(side)
        if g3:
            mod, sign, offset, width = g1.groups()
            if sign == '':
                sign = '+'
            width = g1.groups()[2]
            base = 'd'
        if not (g1 or g2 or g3):
            mod = ''
            sign = '+'
            offset = 0
            width = 0
            base = 'd'
        if base != 'd':
            raise NotImplementedError()
        return (mod, sign, offset, width, base)

    def _generate_line(self):
        """Process one line containing the GENERATE statement from a DNS
        master file."""
        if self.current_origin is None:
            raise UnknownOrigin
        token = self.tok.get()
        try:
            start, stop, step = dns.grange.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except:
            raise dns.exception.SyntaxError
        try:
            lhs = token.value
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except:
            raise dns.exception.SyntaxError
        try:
            ttl = dns.ttl.from_text(token.value)
            self.last_ttl = ttl
            self.last_ttl_known = True
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.ttl.BadTTL:
            if not (self.last_ttl_known or self.default_ttl_known):
                raise dns.exception.SyntaxError('Missing default TTL value')
            if self.default_ttl_known:
                ttl = self.default_ttl
            else:
                ttl = self.last_ttl
        try:
            rdclass = dns.rdataclass.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except dns.exception.SyntaxError:
            raise dns.exception.SyntaxError
        except Exception:
            rdclass = self.zone.rdclass
        if rdclass != self.zone.rdclass:
            raise dns.exception.SyntaxError("RR class is not zone's class")
        try:
            rdtype = dns.rdatatype.from_text(token.value)
            token = self.tok.get()
            if not token.is_identifier():
                raise dns.exception.SyntaxError
        except Exception:
            raise dns.exception.SyntaxError("unknown rdatatype '%s'" % token.value)
        try:
            rhs = token.value
        except:
            raise dns.exception.SyntaxError
        lmod, lsign, loffset, lwidth, lbase = self._parse_modify(lhs)
        rmod, rsign, roffset, rwidth, rbase = self._parse_modify(rhs)
        for i in range(start, stop + 1, step):
            if lsign == u'+':
                lindex = i + int(loffset)
            elif lsign == u'-':
                lindex = i - int(loffset)
            if rsign == u'-':
                rindex = i - int(roffset)
            elif rsign == u'+':
                rindex = i + int(roffset)
            lzfindex = str(lindex).zfill(int(lwidth))
            rzfindex = str(rindex).zfill(int(rwidth))
            name = lhs.replace(u'$%s' % lmod, lzfindex)
            rdata = rhs.replace(u'$%s' % rmod, rzfindex)
            self.last_name = dns.name.from_text(name, self.current_origin)
            name = self.last_name
            if not name.is_subdomain(self.zone.origin):
                self._eat_line()
                return
            if self.relativize:
                name = name.relativize(self.zone.origin)
            n = self.zone.nodes.get(name)
            if n is None:
                n = self.zone.node_factory()
                self.zone.nodes[name] = n
            try:
                rd = dns.rdata.from_text(rdclass, rdtype, rdata, self.current_origin, False)
            except dns.exception.SyntaxError:
                ty, va = sys.exc_info()[:2]
                raise va
            except:
                ty, va = sys.exc_info()[:2]
                raise dns.exception.SyntaxError('caught exception %s: %s' % (str(ty), str(va)))
            rd.choose_relativity(self.zone.origin, self.relativize)
            covers = rd.covers()
            rds = n.find_rdataset(rdclass, rdtype, covers, True)
            rds.add(rd, ttl)

    def read(self):
        """Read a DNS master file and build a zone object.

        @raises dns.zone.NoSOA: No SOA RR was found at the zone origin
        @raises dns.zone.NoNS: No NS RRset was found at the zone origin
        """
        try:
            while 1:
                token = self.tok.get(True, True)
                if token.is_eof():
                    if self.current_file is not None:
                        self.current_file.close()
                    if len(self.saved_state) > 0:
                        self.tok, self.current_origin, self.last_name, self.current_file, self.last_ttl, self.last_ttl_known, self.default_ttl, self.default_ttl_known = self.saved_state.pop(-1)
                        continue
                    break
                elif token.is_eol():
                    continue
                elif token.is_comment():
                    self.tok.get_eol()
                    continue
                elif token.value[0] == u'$':
                    c = token.value.upper()
                    if c == u'$TTL':
                        token = self.tok.get()
                        if not token.is_identifier():
                            raise dns.exception.SyntaxError('bad $TTL')
                        self.default_ttl = dns.ttl.from_text(token.value)
                        self.default_ttl_known = True
                        self.tok.get_eol()
                    elif c == u'$ORIGIN':
                        self.current_origin = self.tok.get_name()
                        self.tok.get_eol()
                        if self.zone.origin is None:
                            self.zone.origin = self.current_origin
                    elif c == u'$INCLUDE' and self.allow_include:
                        token = self.tok.get()
                        filename = token.value
                        token = self.tok.get()
                        if token.is_identifier():
                            new_origin = dns.name.from_text(token.value, self.current_origin)
                            self.tok.get_eol()
                        elif not token.is_eol_or_eof():
                            raise dns.exception.SyntaxError('bad origin in $INCLUDE')
                        else:
                            new_origin = self.current_origin
                        self.saved_state.append((self.tok, self.current_origin, self.last_name, self.current_file, self.last_ttl, self.last_ttl_known, self.default_ttl, self.default_ttl_known))
                        self.current_file = open(filename, 'r')
                        self.tok = dns.tokenizer.Tokenizer(self.current_file, filename)
                        self.current_origin = new_origin
                    elif c == u'$GENERATE':
                        self._generate_line()
                    else:
                        raise dns.exception.SyntaxError("Unknown master file directive '" + c + "'")
                    continue
                self.tok.unget(token)
                self._rr_line()
        except dns.exception.SyntaxError as detail:
            filename, line_number = self.tok.where()
            if detail is None:
                detail = 'syntax error'
            raise dns.exception.SyntaxError('%s:%d: %s' % (filename, line_number, detail))
        if self.check_origin:
            self.zone.check_origin()