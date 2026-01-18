import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def add_change(self, change):
    """ and a new dot point to a changelog entry

        Adds a change entry to the most recent version. The change entry
        should conform to the required format of the changelog (i.e. start
        with two spaces). No line wrapping or anything will be performed,
        so it is advisable to do this yourself if it is a long entry. The
        change will be appended to the current changes, no support is
        provided for per-maintainer changes.
        """
    self._blocks[0].add_change(change)