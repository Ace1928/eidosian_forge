import argparse
import getpass
import inspect
import io
import lzma
import os
import pathlib
import platform
import re
import shutil
import sys
from lzma import CHECK_CRC64, CHECK_SHA256, is_check_supported
from typing import Any, List, Optional
import _lzma  # type: ignore
import multivolumefile
import texttable  # type: ignore
import py7zr
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods
from py7zr.helpers import Local
from py7zr.properties import COMMAND_HELP_STRING
def _run_list(self, target, verbose):
    if not py7zr.is_7zfile(target):
        print('not a 7z file')
        return 1
    with py7zr.SevenZipFile(target, 'r') as a:
        file = sys.stdout
        archive_info = a.archiveinfo()
        archive_list = a.list()
        if verbose:
            if isinstance(target, io.FileIO) or isinstance(target, multivolumefile.MultiVolume):
                file_name: str = target.name
            else:
                file_name = str(target)
            file.write('Listing archive: {}\n'.format(file_name))
            file.write('--\n')
            file.write('Path = {}\n'.format(archive_info.filename))
            file.write('Type = 7z\n')
            fstat = archive_info.stat
            file.write('Phisical Size = {}\n'.format(fstat.st_size))
            file.write('Headers Size = {}\n'.format(archive_info.header_size))
            file.write('Method = {}\n'.format(', '.join(archive_info.method_names)))
            if archive_info.solid:
                file.write('Solid = {}\n'.format('+'))
            else:
                file.write('Solid = {}\n'.format('-'))
            file.write('Blocks = {}\n'.format(archive_info.blocks))
            file.write('\n')
        file.write('total %d files and directories in %sarchive\n' % (len(archive_list), archive_info.solid and 'solid ' or ''))
        file.write('   Date      Time    Attr         Size   Compressed  Name\n')
        file.write('------------------- ----- ------------ ------------  ------------------------\n')
        for f in archive_list:
            if f.creationtime is not None:
                lastwritedate = f.creationtime.astimezone(Local).strftime('%Y-%m-%d')
                lastwritetime = f.creationtime.astimezone(Local).strftime('%H:%M:%S')
            else:
                lastwritedate = '         '
                lastwritetime = '         '
            if f.is_directory:
                attrib = 'D...'
            else:
                attrib = '....'
            if f.archivable:
                attrib += 'A'
            else:
                attrib += '.'
            if f.is_directory:
                extra = '           0 '
            elif f.compressed is None:
                extra = '             '
            else:
                extra = '%12d ' % f.compressed
            file.write('%s %s %s %12d %s %s\n' % (lastwritedate, lastwritetime, attrib, f.uncompressed, extra, f.filename))
        file.write('------------------- ----- ------------ ------------  ------------------------\n')
    return 0