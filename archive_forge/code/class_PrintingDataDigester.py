from __future__ import print_function
import re
import hashlib
class PrintingDataDigester(DataDigester):
    """Extends DataDigester: prints out what we're digesting."""

    def handle_line(self, line):
        print(line.decode('utf8'))
        super(PrintingDataDigester, self).handle_line(line)