import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class MergeRequestBodyParams:
    """Parameter object for the merge_request_body hook."""

    def __init__(self, body, orig_body, directive, to, basename, subject, branch, tree=None):
        self.body = body
        self.orig_body = orig_body
        self.directive = directive
        self.branch = branch
        self.tree = tree
        self.to = to
        self.basename = basename
        self.subject = subject