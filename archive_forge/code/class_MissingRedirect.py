import inspect
import json
import sys
import traceback
class MissingRedirect(CriticalError):
    """At this point in the flow a redirect back to the client was expected."""
    cid = 'missing-redirect'
    msg = 'Expected redirect to the RP, got something else'

    def _func(self, conv=None):
        self._status = self.status
        return {'url': conv.position}