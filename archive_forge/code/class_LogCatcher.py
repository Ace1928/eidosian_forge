import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class LogCatcher(log.LogFormatter):
    """Pull log messages into a list rather than displaying them.

    To simplify testing we save logged revisions here rather than actually
    formatting anything, so that we can precisely check the result without
    being dependent on the formatting.
    """
    supports_merge_revisions = True
    supports_delta = True
    supports_diff = True
    preferred_levels = 0

    def __init__(self, *args, **kwargs):
        kwargs.update(dict(to_file=None))
        super().__init__(*args, **kwargs)
        self.revisions = []

    def log_revision(self, revision):
        self.revisions.append(revision)