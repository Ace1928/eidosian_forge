from . import schema
from .jsonutil import get_column
from .search import Search
def _sub_experiment_values(self, sub_exp, project, experiment_type):
    self._intf._get_entry_point()
    values = []
    column = '%s/%ss/%s/id' % (experiment_type.lower(), sub_exp, sub_exp)
    sub_exps = '%s/experiments?columns=ID,%s' % (self._intf._entry, column)
    if project is not None:
        sub_exps += '&project=%s' % project
    values = get_column(self._get_json(sub_exps), column)
    return list(set(values))