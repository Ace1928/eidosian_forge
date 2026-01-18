import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class CVSGlobDirectoryWalker(GlobDirectoryWalker):
    """An directory tree iterator that checks for CVS data."""

    def filterFiles(self, folder, files):
        """Filters files not listed in CVS subfolder.

        This will look in the CVS subfolder of 'folder' for
        a file named 'Entries' and filter all elements from
        the 'files' list that are not listed in 'Entries'.
        """
        join = os.path.join
        cvsFiles = getCVSEntries(folder)
        if cvsFiles:
            indicesToDelete = []
            for i in range(len(files)):
                f = files[i]
                if join(folder, f) not in cvsFiles:
                    indicesToDelete.append(i)
            indicesToDelete.reverse()
            for i in indicesToDelete:
                del files[i]
        return files