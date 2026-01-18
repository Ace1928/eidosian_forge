import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class RowAccessPolicyReference(Reference):
    _required_fields = frozenset(('projectId', 'datasetId', 'tableId', 'policyId'))
    _format_str = '%(projectId)s:%(datasetId)s.%(tableId)s.%(policyId)s'
    typename = 'row access policy'

    def __init__(self, **kwds):
        self.projectId: str = kwds['projectId']
        self.datasetId: str = kwds['datasetId']
        self.tableId: str = kwds['tableId']
        self.policyId: str = kwds['policyId']
        super().__init__(**kwds)