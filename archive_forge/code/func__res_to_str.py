from __future__ import annotations
from typing import (
def _res_to_str(self, res: rdflib.query.ResultRow, var: str) -> str:
    return '<' + str(res[var]) + '> (' + self._get_local_name(res[var]) + ', ' + str(res['com']) + ')'