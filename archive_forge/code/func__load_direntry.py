from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _load_direntry(self, sid):
    """
        Load a directory entry from the directory.
        This method should only be called once for each storage/stream when
        loading the directory.

        :param sid: index of storage/stream in the directory.
        :returns: a OleDirectoryEntry object

        :exception OleFileError: if the entry has always been referenced.
        """
    if sid < 0 or sid >= len(self.direntries):
        self._raise_defect(DEFECT_FATAL, 'OLE directory index out of range')
    if self.direntries[sid] is not None:
        self._raise_defect(DEFECT_INCORRECT, 'double reference for OLE stream/storage')
        return self.direntries[sid]
    self.directory_fp.seek(sid * 128)
    entry = self.directory_fp.read(128)
    self.direntries[sid] = OleDirectoryEntry(entry, sid, self)
    return self.direntries[sid]