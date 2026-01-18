import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def _load_rsrc_defns(self):
    self._resource_defns = self._template.resource_definitions(self)