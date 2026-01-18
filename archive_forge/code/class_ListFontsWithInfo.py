from Xlib import X
from Xlib.protocol import rq, structs
class ListFontsWithInfo(rq.ReplyRequest):
    _request = rq.Struct(rq.Opcode(50), rq.Pad(1), rq.RequestLength(), rq.Card16('max_names'), rq.LengthOf('pattern', 2), rq.String8('pattern'))
    _reply = rq.Struct(rq.ReplyCode(), rq.LengthOf('name', 1), rq.Card16('sequence_number'), rq.ReplyLength(), rq.Object('min_bounds', structs.CharInfo), rq.Pad(4), rq.Object('max_bounds', structs.CharInfo), rq.Pad(4), rq.Card16('min_char_or_byte2'), rq.Card16('max_char_or_byte2'), rq.Card16('default_char'), rq.LengthOf('properties', 2), rq.Card8('draw_direction'), rq.Card8('min_byte1'), rq.Card8('max_byte1'), rq.Card8('all_chars_exist'), rq.Int16('font_ascent'), rq.Int16('font_descent'), rq.Card32('replies_hint'), rq.List('properties', structs.FontProp), rq.String8('name'))

    def __init__(self, *args, **keys):
        self._fonts = []
        ReplyRequest.__init__(*(self,) + args, **keys)

    def _parse_response(self, data):
        if ord(data[1]) == 0:
            self._response_lock.acquire()
            self._data = self._fonts
            del self._fonts
            self._response_lock.release()
            return
        r, d = self._reply.parse_binary(data)
        self._fonts.append(r)
        self._display.sent_requests.insert(0, self)

    def __getattr__(self, attr):
        raise AttributeError(attr)

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)