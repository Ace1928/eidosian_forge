from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
class PpmPlainDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def _read_block(self):
        return self.fd.read(ImageFile.SAFEBLOCK)

    def _find_comment_end(self, block, start=0):
        a = block.find(b'\n', start)
        b = block.find(b'\r', start)
        return min(a, b) if a * b > 0 else max(a, b)

    def _ignore_comments(self, block):
        if self._comment_spans:
            while block:
                comment_end = self._find_comment_end(block)
                if comment_end != -1:
                    block = block[comment_end + 1:]
                    break
                else:
                    block = self._read_block()
        self._comment_spans = False
        while True:
            comment_start = block.find(b'#')
            if comment_start == -1:
                break
            comment_end = self._find_comment_end(block, comment_start)
            if comment_end != -1:
                block = block[:comment_start] + block[comment_end + 1:]
            else:
                block = block[:comment_start]
                self._comment_spans = True
                break
        return block

    def _decode_bitonal(self):
        """
        This is a separate method because in the plain PBM format, all data tokens are
        exactly one byte, so the inter-token whitespace is optional.
        """
        data = bytearray()
        total_bytes = self.state.xsize * self.state.ysize
        while len(data) != total_bytes:
            block = self._read_block()
            if not block:
                break
            block = self._ignore_comments(block)
            tokens = b''.join(block.split())
            for token in tokens:
                if token not in (48, 49):
                    msg = b'Invalid token for this mode: %s' % bytes([token])
                    raise ValueError(msg)
            data = (data + tokens)[:total_bytes]
        invert = bytes.maketrans(b'01', b'\xff\x00')
        return data.translate(invert)

    def _decode_blocks(self, maxval):
        data = bytearray()
        max_len = 10
        out_byte_count = 4 if self.mode == 'I' else 1
        out_max = 65535 if self.mode == 'I' else 255
        bands = Image.getmodebands(self.mode)
        total_bytes = self.state.xsize * self.state.ysize * bands * out_byte_count
        half_token = False
        while len(data) != total_bytes:
            block = self._read_block()
            if not block:
                if half_token:
                    block = bytearray(b' ')
                else:
                    break
            block = self._ignore_comments(block)
            if half_token:
                block = half_token + block
                half_token = False
            tokens = block.split()
            if block and (not block[-1:].isspace()):
                half_token = tokens.pop()
                if len(half_token) > max_len:
                    msg = b'Token too long found in data: %s' % half_token[:max_len + 1]
                    raise ValueError(msg)
            for token in tokens:
                if len(token) > max_len:
                    msg = b'Token too long found in data: %s' % token[:max_len + 1]
                    raise ValueError(msg)
                value = int(token)
                if value > maxval:
                    msg = f'Channel value too large for this mode: {value}'
                    raise ValueError(msg)
                value = round(value / maxval * out_max)
                data += o32(value) if self.mode == 'I' else o8(value)
                if len(data) == total_bytes:
                    break
        return data

    def decode(self, buffer):
        self._comment_spans = False
        if self.mode == '1':
            data = self._decode_bitonal()
            rawmode = '1;8'
        else:
            maxval = self.args[-1]
            data = self._decode_blocks(maxval)
            rawmode = 'I;32' if self.mode == 'I' else self.mode
        self.set_as_raw(bytes(data), rawmode)
        return (-1, 0)