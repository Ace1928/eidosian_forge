from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _write_mini_stream(self, entry, data_to_write):
    if not entry.sect_chain:
        entry.build_sect_chain(self)
    nb_sectors = len(entry.sect_chain)
    if not self.root.sect_chain:
        self.root.build_sect_chain(self)
    block_size = self.sector_size // self.mini_sector_size
    for idx, sect in enumerate(entry.sect_chain):
        sect_base = sect // block_size
        sect_offset = sect % block_size
        fp_pos = (self.root.sect_chain[sect_base] + 1) * self.sector_size + sect_offset * self.mini_sector_size
        if idx < nb_sectors - 1:
            data_per_sector = data_to_write[idx * self.mini_sector_size:(idx + 1) * self.mini_sector_size]
        else:
            data_per_sector = data_to_write[idx * self.mini_sector_size:]
        self._write_mini_sect(fp_pos, data_per_sector)