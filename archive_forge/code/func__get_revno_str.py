from contextlib import ExitStack
import time
from typing import Type
from breezy import registry
from breezy import revision as _mod_revision
from breezy.osutils import format_date, local_time_offset
def _get_revno_str(self, revision_id):
    numbers = self._branch.revision_id_to_dotted_revno(revision_id)
    revno_str = '.'.join([str(num) for num in numbers])
    return revno_str