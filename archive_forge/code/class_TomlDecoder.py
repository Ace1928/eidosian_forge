import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
class TomlDecoder(object):

    def __init__(self, _dict=dict):
        self._dict = _dict

    def get_empty_table(self):
        return self._dict()

    def get_empty_inline_table(self):

        class DynamicInlineTableDict(self._dict, InlineTableDict):
            """Concrete sentinel subclass for inline tables.
            It is a subclass of _dict which is passed in dynamically at load
            time

            It is also a subclass of InlineTableDict
            """
        return DynamicInlineTableDict()

    def load_inline_object(self, line, currentlevel, multikey=False, multibackslash=False):
        candidate_groups = line[1:-1].split(',')
        groups = []
        if len(candidate_groups) == 1 and (not candidate_groups[0].strip()):
            candidate_groups.pop()
        while len(candidate_groups) > 0:
            candidate_group = candidate_groups.pop(0)
            try:
                _, value = candidate_group.split('=', 1)
            except ValueError:
                raise ValueError('Invalid inline table encountered')
            value = value.strip()
            if value[0] == value[-1] and value[0] in ('"', "'") or (value[0] in '-0123456789' or value in ('true', 'false') or (value[0] == '[' and value[-1] == ']') or (value[0] == '{' and value[-1] == '}')):
                groups.append(candidate_group)
            elif len(candidate_groups) > 0:
                candidate_groups[0] = candidate_group + ',' + candidate_groups[0]
            else:
                raise ValueError('Invalid inline table value encountered')
        for group in groups:
            status = self.load_line(group, currentlevel, multikey, multibackslash)
            if status is not None:
                break

    def _get_split_on_quotes(self, line):
        doublequotesplits = line.split('"')
        quoted = False
        quotesplits = []
        if len(doublequotesplits) > 1 and "'" in doublequotesplits[0]:
            singlequotesplits = doublequotesplits[0].split("'")
            doublequotesplits = doublequotesplits[1:]
            while len(singlequotesplits) % 2 == 0 and len(doublequotesplits):
                singlequotesplits[-1] += '"' + doublequotesplits[0]
                doublequotesplits = doublequotesplits[1:]
                if "'" in singlequotesplits[-1]:
                    singlequotesplits = singlequotesplits[:-1] + singlequotesplits[-1].split("'")
            quotesplits += singlequotesplits
        for doublequotesplit in doublequotesplits:
            if quoted:
                quotesplits.append(doublequotesplit)
            else:
                quotesplits += doublequotesplit.split("'")
                quoted = not quoted
        return quotesplits

    def load_line(self, line, currentlevel, multikey, multibackslash):
        i = 1
        quotesplits = self._get_split_on_quotes(line)
        quoted = False
        for quotesplit in quotesplits:
            if not quoted and '=' in quotesplit:
                break
            i += quotesplit.count('=')
            quoted = not quoted
        pair = line.split('=', i)
        strictly_valid = _strictly_valid_num(pair[-1])
        if _number_with_underscores.match(pair[-1]):
            pair[-1] = pair[-1].replace('_', '')
        while len(pair[-1]) and (pair[-1][0] != ' ' and pair[-1][0] != '\t' and (pair[-1][0] != "'") and (pair[-1][0] != '"') and (pair[-1][0] != '[') and (pair[-1][0] != '{') and (pair[-1].strip() != 'true') and (pair[-1].strip() != 'false')):
            try:
                float(pair[-1])
                break
            except ValueError:
                pass
            if _load_date(pair[-1]) is not None:
                break
            if TIME_RE.match(pair[-1]):
                break
            i += 1
            prev_val = pair[-1]
            pair = line.split('=', i)
            if prev_val == pair[-1]:
                raise ValueError('Invalid date or number')
            if strictly_valid:
                strictly_valid = _strictly_valid_num(pair[-1])
        pair = ['='.join(pair[:-1]).strip(), pair[-1].strip()]
        if '.' in pair[0]:
            if '"' in pair[0] or "'" in pair[0]:
                quotesplits = self._get_split_on_quotes(pair[0])
                quoted = False
                levels = []
                for quotesplit in quotesplits:
                    if quoted:
                        levels.append(quotesplit)
                    else:
                        levels += [level.strip() for level in quotesplit.split('.')]
                    quoted = not quoted
            else:
                levels = pair[0].split('.')
            while levels[-1] == '':
                levels = levels[:-1]
            for level in levels[:-1]:
                if level == '':
                    continue
                if level not in currentlevel:
                    currentlevel[level] = self.get_empty_table()
                currentlevel = currentlevel[level]
            pair[0] = levels[-1].strip()
        elif (pair[0][0] == '"' or pair[0][0] == "'") and pair[0][-1] == pair[0][0]:
            pair[0] = _unescape(pair[0][1:-1])
        k, koffset = self._load_line_multiline_str(pair[1])
        if k > -1:
            while k > -1 and pair[1][k + koffset] == '\\':
                multibackslash = not multibackslash
                k -= 1
            if multibackslash:
                multilinestr = pair[1][:-1]
            else:
                multilinestr = pair[1] + '\n'
            multikey = pair[0]
        else:
            value, vtype = self.load_value(pair[1], strictly_valid)
        try:
            currentlevel[pair[0]]
            raise ValueError('Duplicate keys!')
        except TypeError:
            raise ValueError('Duplicate keys!')
        except KeyError:
            if multikey:
                return (multikey, multilinestr, multibackslash)
            else:
                currentlevel[pair[0]] = value

    def _load_line_multiline_str(self, p):
        poffset = 0
        if len(p) < 3:
            return (-1, poffset)
        if p[0] == '[' and (p.strip()[-1] != ']' and self._load_array_isstrarray(p)):
            newp = p[1:].strip().split(',')
            while len(newp) > 1 and newp[-1][0] != '"' and (newp[-1][0] != "'"):
                newp = newp[:-2] + [newp[-2] + ',' + newp[-1]]
            newp = newp[-1]
            poffset = len(p) - len(newp)
            p = newp
        if p[0] != '"' and p[0] != "'":
            return (-1, poffset)
        if p[1] != p[0] or p[2] != p[0]:
            return (-1, poffset)
        if len(p) > 5 and p[-1] == p[0] and (p[-2] == p[0]) and (p[-3] == p[0]):
            return (-1, poffset)
        return (len(p) - 1, poffset)

    def load_value(self, v, strictly_valid=True):
        if not v:
            raise ValueError('Empty value is invalid')
        if v == 'true':
            return (True, 'bool')
        elif v.lower() == 'true':
            raise ValueError('Only all lowercase booleans allowed')
        elif v == 'false':
            return (False, 'bool')
        elif v.lower() == 'false':
            raise ValueError('Only all lowercase booleans allowed')
        elif v[0] == '"' or v[0] == "'":
            quotechar = v[0]
            testv = v[1:].split(quotechar)
            triplequote = False
            triplequotecount = 0
            if len(testv) > 1 and testv[0] == '' and (testv[1] == ''):
                testv = testv[2:]
                triplequote = True
            closed = False
            for tv in testv:
                if tv == '':
                    if triplequote:
                        triplequotecount += 1
                    else:
                        closed = True
                else:
                    oddbackslash = False
                    try:
                        i = -1
                        j = tv[i]
                        while j == '\\':
                            oddbackslash = not oddbackslash
                            i -= 1
                            j = tv[i]
                    except IndexError:
                        pass
                    if not oddbackslash:
                        if closed:
                            raise ValueError('Found tokens after a closed ' + 'string. Invalid TOML.')
                        elif not triplequote or triplequotecount > 1:
                            closed = True
                        else:
                            triplequotecount = 0
            if quotechar == '"':
                escapeseqs = v.split('\\')[1:]
                backslash = False
                for i in escapeseqs:
                    if i == '':
                        backslash = not backslash
                    else:
                        if i[0] not in _escapes and (i[0] != 'u' and i[0] != 'U' and (not backslash)):
                            raise ValueError('Reserved escape sequence used')
                        if backslash:
                            backslash = False
                for prefix in ['\\u', '\\U']:
                    if prefix in v:
                        hexbytes = v.split(prefix)
                        v = _load_unicode_escapes(hexbytes[0], hexbytes[1:], prefix)
                v = _unescape(v)
            if len(v) > 1 and v[1] == quotechar and (len(v) < 3 or v[1] == v[2]):
                v = v[2:-2]
            return (v[1:-1], 'str')
        elif v[0] == '[':
            return (self.load_array(v), 'array')
        elif v[0] == '{':
            inline_object = self.get_empty_inline_table()
            self.load_inline_object(v, inline_object)
            return (inline_object, 'inline_object')
        elif TIME_RE.match(v):
            h, m, s, _, ms = TIME_RE.match(v).groups()
            time = datetime.time(int(h), int(m), int(s), int(ms) if ms else 0)
            return (time, 'time')
        else:
            parsed_date = _load_date(v)
            if parsed_date is not None:
                return (parsed_date, 'date')
            if not strictly_valid:
                raise ValueError('Weirdness with leading zeroes or underscores in your number.')
            itype = 'int'
            neg = False
            if v[0] == '-':
                neg = True
                v = v[1:]
            elif v[0] == '+':
                v = v[1:]
            v = v.replace('_', '')
            lowerv = v.lower()
            if '.' in v or ('x' not in v and ('e' in v or 'E' in v)):
                if '.' in v and v.split('.', 1)[1] == '':
                    raise ValueError('This float is missing digits after the point')
                if v[0] not in '0123456789':
                    raise ValueError("This float doesn't have a leading digit")
                v = float(v)
                itype = 'float'
            elif len(lowerv) == 3 and (lowerv == 'inf' or lowerv == 'nan'):
                v = float(v)
                itype = 'float'
            if itype == 'int':
                v = int(v, 0)
            if neg:
                return (0 - v, itype)
            return (v, itype)

    def bounded_string(self, s):
        if len(s) == 0:
            return True
        if s[-1] != s[0]:
            return False
        i = -2
        backslash = False
        while len(s) + i > 0:
            if s[i] == '\\':
                backslash = not backslash
                i -= 1
            else:
                break
        return not backslash

    def _load_array_isstrarray(self, a):
        a = a[1:-1].strip()
        if a != '' and (a[0] == '"' or a[0] == "'"):
            return True
        return False

    def load_array(self, a):
        atype = None
        retval = []
        a = a.strip()
        if '[' not in a[1:-1] or '' != a[1:-1].split('[')[0].strip():
            strarray = self._load_array_isstrarray(a)
            if not a[1:-1].strip().startswith('{'):
                a = a[1:-1].split(',')
            else:
                new_a = []
                start_group_index = 1
                end_group_index = 2
                open_bracket_count = 1 if a[start_group_index] == '{' else 0
                in_str = False
                while end_group_index < len(a[1:]):
                    if a[end_group_index] == '"' or a[end_group_index] == "'":
                        if in_str:
                            backslash_index = end_group_index - 1
                            while backslash_index > -1 and a[backslash_index] == '\\':
                                in_str = not in_str
                                backslash_index -= 1
                        in_str = not in_str
                    if not in_str and a[end_group_index] == '{':
                        open_bracket_count += 1
                    if in_str or a[end_group_index] != '}':
                        end_group_index += 1
                        continue
                    elif a[end_group_index] == '}' and open_bracket_count > 1:
                        open_bracket_count -= 1
                        end_group_index += 1
                        continue
                    end_group_index += 1
                    new_a.append(a[start_group_index:end_group_index])
                    start_group_index = end_group_index + 1
                    while start_group_index < len(a[1:]) and a[start_group_index] != '{':
                        start_group_index += 1
                    end_group_index = start_group_index + 1
                a = new_a
            b = 0
            if strarray:
                while b < len(a) - 1:
                    ab = a[b].strip()
                    while not self.bounded_string(ab) or (len(ab) > 2 and ab[0] == ab[1] == ab[2] and (ab[-2] != ab[0]) and (ab[-3] != ab[0])):
                        a[b] = a[b] + ',' + a[b + 1]
                        ab = a[b].strip()
                        if b < len(a) - 2:
                            a = a[:b + 1] + a[b + 2:]
                        else:
                            a = a[:b + 1]
                    b += 1
        else:
            al = list(a[1:-1])
            a = []
            openarr = 0
            j = 0
            for i in _range(len(al)):
                if al[i] == '[':
                    openarr += 1
                elif al[i] == ']':
                    openarr -= 1
                elif al[i] == ',' and (not openarr):
                    a.append(''.join(al[j:i]))
                    j = i + 1
            a.append(''.join(al[j:]))
        for i in _range(len(a)):
            a[i] = a[i].strip()
            if a[i] != '':
                nval, ntype = self.load_value(a[i])
                if atype:
                    if ntype != atype:
                        raise ValueError('Not a homogeneous array')
                else:
                    atype = ntype
                retval.append(nval)
        return retval

    def preserve_comment(self, line_no, key, comment, beginline):
        pass

    def embed_comments(self, idx, currentlevel):
        pass