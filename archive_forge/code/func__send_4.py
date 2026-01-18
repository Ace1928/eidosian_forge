import os
import time
from typing import Callable
from . import controldir, errors, osutils, registry, trace
from .branch import Branch
from .i18n import gettext
from .revision import NULL_REVISION
def _send_4(branch, revision_id, target_branch, public_branch, no_patch, no_bundle, message, base_revision_id, local_target_branch=None):
    from breezy import merge_directive
    return merge_directive.MergeDirective2.from_objects(branch.repository, revision_id, time.time(), osutils.local_time_offset(), target_branch, public_branch=public_branch, include_patch=not no_patch, include_bundle=not no_bundle, message=message, base_revision_id=base_revision_id, local_target_branch=local_target_branch)