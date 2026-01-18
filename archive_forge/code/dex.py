#
# diffoscope: in-depth comparison of files, archives, and directories
#
# Copyright © 2015 Reiner Herrmann <reiner@reiner-h.de>
# Copyright © 2016-2020 Chris Lamb <lamby@debian.org>
#
# diffoscope is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffoscope is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffoscope.  If not, see <https://www.gnu.org/licenses/>.

import re
import os.path
import logging
import subprocess

from diffoscope.difference import Difference
from diffoscope.tools import tool_required, tool_check_installed

from .utils.file import File
from .utils.archive import Archive
from .utils.command import Command

logger = logging.getLogger(__name__)


class DexContainer(Archive):
    @property
    def path(self):
        return self._path

    def open_archive(self):
        return self

    def close_archive(self):
        pass

    def get_member_names(self):
        return [self.get_compressed_content_name(".dex") + ".jar"]

    @tool_required("enjarify")
    def extract(self, member_name, dest_dir):
        dest_path = os.path.join(dest_dir, member_name)
        logger.debug("dex extracting to %s", dest_path)
        subprocess.check_call(
            ["enjarify", "-o", dest_path, self.source.path],
            stderr=None,
            stdout=subprocess.PIPE,
        )
        return dest_path


class DexFile(File):
    DESCRIPTION = "Dalvik .dex files"
    FILE_TYPE_HEADER_PREFIX = b"dex\n"
    FILE_TYPE_RE = re.compile(r"^Dalvik dex file\b")
    FILE_EXTENSION_SUFFIX = {".dex"}
    CONTAINER_CLASSES = [DexContainer]

    def compare_details(self, other, source=None):
        if tool_check_installed("dexdump"):
            dexdump = Difference.from_operation(Dexdump, self.path, other.path)
            if dexdump is not None:
                dexdump.add_comment(
                    "Ignoring differences in offsets to keep diff size reasonable."
                )
                return [dexdump]
        else:
            self.add_comment(
                "'dexdump' command not installed; cannot compare format-specific details."
            )
        return []


class Dexdump(Command):
    @tool_required("dexdump")
    def cmdline(self):
        # -a : display annotations
        # -d : disassemble code sections
        # -f : display summary information from file header
        # -h : display file header details
        return ["dexdump", "-a", "-d", "-f", "-h", self.path]

    def filter(self, line):
        # Processing 'classes.dex'...
        if line.startswith(b"Processing "):
            return b""

        # Opened 'classes.dex', DEX version '037'
        if line.startswith(b"Opened "):
            return line.split(b", ", 1)[-1]

        # ignore differences in offsets
        offsets = (b"annotations", b"class_data", b"interfaces")
        if any(line.startswith(o + b"_off") for o in offsets):
            return b""

        return line
