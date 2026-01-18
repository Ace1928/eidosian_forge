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
@six.add_metaclass(abc.ABCMeta)
class Updater(BaseUpdater):
    """A resource cache table updater.

  An updater returns a list of parsed parameter tuples that replaces the rows in
  one cache table. It can also adjust the table timeout.

  The parameters may have their own updaters. These objects are organized as a
  tree with one resource at the root.

  Attributes:
    cache: The persistent cache object.
    collection: The resource collection name.
    columns: The number of columns in the parsed resource parameter tuple.
    parameters: A list of Parameter objects.
    timeout: The resource table timeout in seconds, 0 for no timeout (0 is easy
      to represent in a persistent cache tuple which holds strings and numbers).
  """

    def __init__(self, cache=None, collection=None, columns=0, column=0, parameters=None, timeout=DEFAULT_TIMEOUT):
        """Updater constructor.

    Args:
      cache: The persistent cache object.
      collection: The resource collection name that (1) uniquely names the
        table(s) for the parsed resource parameters (2) is the lookup name of
        the resource URI parser. Resource collection names are unique by
        definition. Non-resource collection names must not clash with resource
        collections names. Prepending a '.' to non-resource collections names
        will avoid the clash.
      columns: The number of columns in the parsed resource parameter tuple.
        Must be >= 1.
      column: If this is an updater for an aggregate parameter then the updater
        produces a table of aggregate_resource tuples. The parent collection
        copies aggregate_resource[column] to a column in its own resource
        parameter tuple.
      parameters: A list of Parameter objects.
      timeout: The resource table timeout in seconds, 0 for no timeout.
    """
        super(Updater, self).__init__()
        self.cache = cache
        self.collection = collection
        self.columns = columns if collection else 1
        self.column = column
        self.parameters = parameters or []
        self.timeout = timeout or 0

    def _GetTableName(self, suffix_list=None):
        """Returns the table name; the module path if no collection.

    Args:
      suffix_list: a list of values to attach to the end of the table name.
        Typically, these will be aggregator values, like project ID.
    Returns: a name to use for the table in the cache DB.
    """
        if self.collection:
            name = [self.collection]
        else:
            name = [module_util.GetModulePath(self)]
        if suffix_list:
            name.extend(suffix_list)
        return '.'.join(name)

    def _GetRuntimeParameters(self, parameter_info):
        """Constructs and returns the _RuntimeParameter list.

    This method constructs a muable shadow of self.parameters with updater_class
    and table instantiations. Each runtime parameter can be:

    (1) A static value derived from parameter_info.
    (2) A parameter with it's own updater_class.  The updater is used to list
        all of the possible values for the parameter.
    (3) An unknown value (None).  The possible values are contained in the
        resource cache for self.

    The Select method combines the caller supplied row template and the runtime
    parameters to filter the list of parsed resources in the resource cache.

    Args:
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.

    Returns:
      The runtime parameters shadow of the immutable self.parameters.
    """
        runtime_parameters = []
        for parameter in self.parameters:
            updater_class, aggregator = parameter_info.GetUpdater(parameter.name)
            value = parameter_info.GetValue(parameter.name, check_properties=aggregator)
            runtime_parameter = _RuntimeParameter(parameter, updater_class, value, aggregator)
            runtime_parameters.append(runtime_parameter)
        return runtime_parameters

    def ParameterInfo(self):
        """Returns the parameter info object."""
        return ParameterInfo()

    def SelectTable(self, table, row_template, parameter_info, aggregations=None):
        """Returns the list of rows matching row_template in table.

    Refreshes expired tables by calling the updater.

    Args:
      table: The persistent table object.
      row_template: A row template to match in Select().
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.
      aggregations: A list of aggregation Parameter objects.

    Returns:
      The list of rows matching row_template in table.
    """
        if not aggregations:
            aggregations = []
        log.info('cache table=%s aggregations=[%s]', table.name, ' '.join(['{}={}'.format(x.name, x.value) for x in aggregations]))
        try:
            return table.Select(row_template)
        except exceptions.CacheTableExpired:
            rows = self.Update(parameter_info, aggregations)
            if rows is not None:
                table.DeleteRows()
                table.AddRows(rows)
                table.Validate()
            return table.Select(row_template, ignore_expiration=True)

    def Select(self, row_template, parameter_info=None):
        """Returns the list of rows matching row_template in the collection.

    All tables in the collection are in play. The row matching done by the
    cache layer conveniently prunes the number of tables accessed.

    Args:
      row_template: A row template tuple. The number of columns in the template
        must match the number of columns in the collection. A column with value
        None means match all values for the column. Each column may contain
        these wildcard characters:
          * - match any string of zero or more characters
          ? - match any character
        The matching is anchored on the left.
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.

    Returns:
      The list of rows that match the template row.
    """
        template = list(row_template)
        if self.columns > len(template):
            template += [None] * (self.columns - len(template))
        log.info('cache template=[%s]', ', '.join(["'{}'".format(t) for t in template]))
        values = [[]]
        aggregations = []
        parameters = self._GetRuntimeParameters(parameter_info)
        for i, parameter in enumerate(parameters):
            parameter.generate = False
            if parameter.value and template[parameter.column] in (None, '*'):
                template[parameter.column] = parameter.value
                log.info('cache parameter=%s column=%s value=%s aggregate=%s', parameter.name, parameter.column, parameter.value, parameter.aggregator)
                if parameter.aggregator:
                    aggregations.append(parameter)
                    parameter.generate = True
                    for v in values:
                        v.append(parameter.value)
            elif parameter.aggregator:
                aggregations.append(parameter)
                parameter.generate = True
                log.info('cache parameter=%s column=%s value=%s aggregate=%s', parameter.name, parameter.column, parameter.value, parameter.aggregator)
                updater = parameter.updater_class(cache=self.cache)
                sub_template = [None] * updater.columns
                sub_template[updater.column] = template[parameter.column]
                log.info('cache parameter=%s column=%s aggregate=%s', parameter.name, parameter.column, parameter.aggregator)
                new_values = []
                for perm, selected in updater.YieldSelectTableFromPermutations(parameters[:i], values, sub_template, parameter_info):
                    updater.ExtendValues(new_values, perm, selected)
                values = new_values
        if not values:
            aggregation_values = [x.value for x in aggregations]
            if None in aggregation_values:
                return []
            table_name = self._GetTableName(suffix_list=aggregation_values)
            table = self.cache.Table(table_name, columns=self.columns, keys=self.columns, timeout=self.timeout)
            return self.SelectTable(table, template, parameter_info, aggregations)
        rows = []
        for _, selected in self.YieldSelectTableFromPermutations(parameters, values, template, parameter_info):
            rows.extend(selected)
        log.info('cache rows=%s' % rows)
        return rows

    def _GetParameterColumn(self, parameter_info, parameter_name):
        """Get this updater's column number for a certain parameter."""
        updater_parameters = self._GetRuntimeParameters(parameter_info)
        for parameter in updater_parameters:
            if parameter.name == parameter_name:
                return parameter.column
        return None

    def ExtendValues(self, values, perm, selected):
        """Add selected values to a template and extend the selected rows."""
        vals = [row[self.column] for row in selected]
        log.info('cache collection={} adding values={}'.format(self.collection, vals))
        v = [perm + [val] for val in vals]
        values.extend(v)

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

    def GetTableForRow(self, row, parameter_info=None, create=True):
        """Returns the table for row.

    Args:
      row: The fully populated resource row.
      parameter_info: A ParamaterInfo object for accessing parameter values in
        the program state.
      create: Create the table if it doesn't exist if True.

    Returns:
      The table for row.
    """
        parameters = self._GetRuntimeParameters(parameter_info)
        values = [row[p.column] for p in parameters if p.aggregator]
        return self.cache.Table(self._GetTableName(suffix_list=values), columns=self.columns, keys=self.columns, timeout=self.timeout, create=create)

    @abc.abstractmethod
    def Update(self, parameter_info=None, aggregations=None):
        """Returns the list of all current parsed resource parameters."""
        del parameter_info, aggregations