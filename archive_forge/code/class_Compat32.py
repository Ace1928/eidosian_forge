import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
@_extend_docstrings
class Compat32(Policy):
    """+
    This particular policy is the backward compatibility Policy.  It
    replicates the behavior of the email package version 5.1.
    """
    mangle_from_ = True

    def _sanitize_header(self, name, value):
        if not isinstance(value, str):
            return value
        if _has_surrogates(value):
            return header.Header(value, charset=_charset.UNKNOWN8BIT, header_name=name)
        else:
            return value

    def header_source_parse(self, sourcelines):
        """+
        The name is parsed as everything up to the ':' and returned unmodified.
        The value is determined by stripping leading whitespace off the
        remainder of the first line, joining all subsequent lines together, and
        stripping any trailing carriage return or linefeed characters.

        """
        name, value = sourcelines[0].split(':', 1)
        value = value.lstrip(' \t') + ''.join(sourcelines[1:])
        return (name, value.rstrip('\r\n'))

    def header_store_parse(self, name, value):
        """+
        The name and value are returned unmodified.
        """
        return (name, value)

    def header_fetch_parse(self, name, value):
        """+
        If the value contains binary data, it is converted into a Header object
        using the unknown-8bit charset.  Otherwise it is returned unmodified.
        """
        return self._sanitize_header(name, value)

    def fold(self, name, value):
        """+
        Headers are folded using the Header folding algorithm, which preserves
        existing line breaks in the value, and wraps each resulting line to the
        max_line_length.  Non-ASCII binary data are CTE encoded using the
        unknown-8bit charset.

        """
        return self._fold(name, value, sanitize=True)

    def fold_binary(self, name, value):
        """+
        Headers are folded using the Header folding algorithm, which preserves
        existing line breaks in the value, and wraps each resulting line to the
        max_line_length.  If cte_type is 7bit, non-ascii binary data is CTE
        encoded using the unknown-8bit charset.  Otherwise the original source
        header is used, with its existing line breaks and/or binary data.

        """
        folded = self._fold(name, value, sanitize=self.cte_type == '7bit')
        return folded.encode('ascii', 'surrogateescape')

    def _fold(self, name, value, sanitize):
        parts = []
        parts.append('%s: ' % name)
        if isinstance(value, str):
            if _has_surrogates(value):
                if sanitize:
                    h = header.Header(value, charset=_charset.UNKNOWN8BIT, header_name=name)
                else:
                    parts.append(value)
                    h = None
            else:
                h = header.Header(value, header_name=name)
        else:
            h = value
        if h is not None:
            maxlinelen = 0
            if self.max_line_length is not None:
                maxlinelen = self.max_line_length
            parts.append(h.encode(linesep=self.linesep, maxlinelen=maxlinelen))
        parts.append(self.linesep)
        return ''.join(parts)