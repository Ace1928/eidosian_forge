import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
class AlwaysRevert(Retry):
    """Retry that always reverts subflow."""

    def on_failure(self, *args, **kwargs):
        return REVERT

    def execute(self, *args, **kwargs):
        pass