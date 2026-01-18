from __future__ import annotations
import io
import struct
from . import Image, ImageFile
from ._binary import i16le as i16
from ._binary import o16le as o16
class MspDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        img = io.BytesIO()
        blank_line = bytearray((255,) * ((self.state.xsize + 7) // 8))
        try:
            self.fd.seek(32)
            rowmap = struct.unpack_from(f'<{self.state.ysize}H', self.fd.read(self.state.ysize * 2))
        except struct.error as e:
            msg = 'Truncated MSP file in row map'
            raise OSError(msg) from e
        for x, rowlen in enumerate(rowmap):
            try:
                if rowlen == 0:
                    img.write(blank_line)
                    continue
                row = self.fd.read(rowlen)
                if len(row) != rowlen:
                    msg = f'Truncated MSP file, expected {rowlen} bytes on row {x}'
                    raise OSError(msg)
                idx = 0
                while idx < rowlen:
                    runtype = row[idx]
                    idx += 1
                    if runtype == 0:
                        runcount, runval = struct.unpack_from('Bc', row, idx)
                        img.write(runval * runcount)
                        idx += 2
                    else:
                        runcount = runtype
                        img.write(row[idx:idx + runcount])
                        idx += runcount
            except struct.error as e:
                msg = f'Corrupted MSP file in row {x}'
                raise OSError(msg) from e
        self.set_as_raw(img.getvalue(), ('1', 0, 1))
        return (-1, 0)