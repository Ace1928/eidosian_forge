import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
class Yaql(function.Function):
    """A function for executing a yaql expression.

    Takes the form::

        yaql:
            expression:
                <body>
            data:
                <var>: <list>

    Evaluates expression <body> on the given data.
    """
    _parser = None

    @classmethod
    def get_yaql_parser(cls):
        if cls._parser is None:
            global_options = {'yaql.limitIterators': cfg.CONF.yaql.limit_iterators, 'yaql.memoryQuota': cfg.CONF.yaql.memory_quota}
            cls._parser = yaql.YaqlFactory().create(global_options)
            cls._context = yaql.create_context()
        return cls._parser

    def __init__(self, stack, fn_name, args):
        super(Yaql, self).__init__(stack, fn_name, args)
        if not isinstance(self.args, collections.abc.Mapping):
            raise TypeError(_('Arguments to "%s" must be a map.') % self.fn_name)
        try:
            self._expression = self.args['expression']
            self._data = self.args.get('data', {})
            if set(self.args) - set(['expression', 'data']):
                raise KeyError
        except (KeyError, TypeError):
            example = '%s:\n              expression: $.data.var1.sum()\n              data:\n                var1: [3, 2, 1]' % self.fn_name
            raise KeyError(_('"%(name)s" syntax should be %(example)s') % {'name': self.fn_name, 'example': example})

    def validate(self):
        super(Yaql, self).validate()
        if not isinstance(self._expression, function.Function):
            self._parse(self._expression)

    def _parse(self, expression):
        if not isinstance(expression, str):
            raise TypeError(_('The "expression" argument to %s must contain a string.') % self.fn_name)
        parse = self.get_yaql_parser()
        try:
            return parse(expression)
        except exceptions.YaqlException as yex:
            raise ValueError(_('Bad expression %s.') % yex)

    def result(self):
        statement = self._parse(function.resolve(self._expression))
        data = function.resolve(self._data)
        context = self._context.create_child_context()
        return statement.evaluate({'data': data}, context)