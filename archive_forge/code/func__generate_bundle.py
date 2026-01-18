import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
@staticmethod
def _generate_bundle(repository, revision_id, ancestor_id):
    s = BytesIO()
    bundle_serializer.write_bundle(repository, revision_id, ancestor_id, s, '0.9')
    return s.getvalue()