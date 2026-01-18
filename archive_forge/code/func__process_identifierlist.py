from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_identifierlist(self, tlist):
    identifiers = list(tlist.get_identifiers())
    if self.indent_columns:
        first = next(identifiers[0].flatten())
        num_offset = 1 if self.char == '\t' else self.width
    else:
        first = next(identifiers.pop(0).flatten())
        num_offset = 1 if self.char == '\t' else self._get_offset(first)
    if not tlist.within(sql.Function) and (not tlist.within(sql.Values)):
        with offset(self, num_offset):
            position = 0
            for token in identifiers:
                position += len(token.value) + 1
                if position > self.wrap_after - self.offset:
                    adjust = 0
                    if self.comma_first:
                        adjust = -2
                        _, comma = tlist.token_prev(tlist.token_index(token))
                        if comma is None:
                            continue
                        token = comma
                    tlist.insert_before(token, self.nl(offset=adjust))
                    if self.comma_first:
                        _, ws = tlist.token_next(tlist.token_index(token), skip_ws=False)
                        if ws is not None and ws.ttype is not T.Text.Whitespace:
                            tlist.insert_after(token, sql.Token(T.Whitespace, ' '))
                    position = 0
    else:
        for token in tlist:
            _, next_ws = tlist.token_next(tlist.token_index(token), skip_ws=False)
            if token.value == ',' and (not next_ws.is_whitespace):
                tlist.insert_after(token, sql.Token(T.Whitespace, ' '))
        end_at = self.offset + sum((len(i.value) + 1 for i in identifiers))
        adjusted_offset = 0
        if self.wrap_after > 0 and end_at > self.wrap_after - self.offset and self._last_func:
            adjusted_offset = -len(self._last_func.value) - 1
        with offset(self, adjusted_offset), indent(self):
            if adjusted_offset < 0:
                tlist.insert_before(identifiers[0], self.nl())
            position = 0
            for token in identifiers:
                position += len(token.value) + 1
                if self.wrap_after > 0 and position > self.wrap_after - self.offset:
                    adjust = 0
                    tlist.insert_before(token, self.nl(offset=adjust))
                    position = 0
    self._process_default(tlist)