import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class CustomLogFormatter(log.LogFormatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.revisions = []

    def get_levels(self):
        return 0

    def log_revision(self, revision):
        self.revisions.append(revision)