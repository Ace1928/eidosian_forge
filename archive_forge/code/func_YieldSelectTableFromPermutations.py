from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import file_cache
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def YieldSelectTableFromPermutations(self, parameters, values, template, parameter_info):
    """Selects completions from tables using multiple permutations of values.

    For each vector in values, e.g. ['my-project', 'my-zone'], this method
    selects rows matching the template from a leaf table corresponding to the
    vector (e.g. 'my.collection.my-project.my-zone') and yields a 2-tuple
    containing that vector and the selected rows.

    Args:
      parameters: [Parameter], the list of parameters up through the
        current updater belonging to the parent. These will be used to iterate
        through each permutation contained in values.
      values: list(list()), a list of lists of valid values. Each item in values
        corresponds to a single permutation of values for which item[n] is a
        possible value for the nth generator in parent_parameters.
      template: list(str), the template to use to select new values.
      parameter_info: ParameterInfo, the object that is used to get runtime
        values.

    Yields:
      (perm, list(list)): a 2-tuple where the first value is the permutation
        currently being used to select values and the second value is the result
        of selecting to match the permutation.
    """
    for perm in values:
        temp_perm = [val for val in perm]
        table = self.cache.Table(self._GetTableName(suffix_list=perm), columns=self.columns, keys=self.columns, timeout=self.timeout)
        aggregations = []
        for parameter in parameters:
            if parameter.generate:
                column = self._GetParameterColumn(parameter_info, parameter.name)
                if column is None:
                    continue
                template[column] = temp_perm.pop(0)
                parameter.value = template[column]
            if parameter.value:
                aggregations.append(parameter)
        selected = self.SelectTable(table, template, parameter_info, aggregations)
        yield (perm, selected)