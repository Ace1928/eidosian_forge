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
class GetAttThenSelect(function.Function):
    """A function for resolving resource attributes.

    Takes the form::

        get_attr:
          - <resource_name>
          - <attribute_name>
          - <path1>
          - ...
    """

    def __init__(self, stack, fn_name, args):
        super(GetAttThenSelect, self).__init__(stack, fn_name, args)
        self._resource_name, self._attribute, self._path_components = self._parse_args()

    def _parse_args(self):
        if not isinstance(self.args, collections.abc.Sequence) or isinstance(self.args, str):
            raise TypeError(_('Argument to "%s" must be a list') % self.fn_name)
        if len(self.args) < 2:
            raise ValueError(_('Arguments to "%s" must be of the form [resource_name, attribute, (path), ...]') % self.fn_name)
        return (self.args[0], self.args[1], self.args[2:])

    def _res_name(self):
        return function.resolve(self._resource_name)

    def _resource(self, path='unknown'):
        resource_name = self._res_name()
        try:
            return self.stack[resource_name]
        except KeyError:
            raise exception.InvalidTemplateReference(resource=resource_name, key=path)

    def _attr_path(self):
        return function.resolve(self._attribute)

    def dep_attrs(self, resource_name):
        if self._res_name() == resource_name:
            try:
                attrs = [self._attr_path()]
            except Exception as exc:
                LOG.debug('Ignoring exception calculating required attributes: %s %s', type(exc).__name__, str(exc))
                attrs = []
        else:
            attrs = []
        return itertools.chain(super(GetAttThenSelect, self).dep_attrs(resource_name), attrs)

    def all_dep_attrs(self):
        try:
            attrs = [(self._res_name(), self._attr_path())]
        except Exception:
            attrs = []
        return itertools.chain(function.all_dep_attrs(self.args), attrs)

    def dependencies(self, path):
        return itertools.chain(super(GetAttThenSelect, self).dependencies(path), [self._resource(path)])

    def _allow_without_attribute_name(self):
        return False

    def validate(self):
        super(GetAttThenSelect, self).validate()
        res = self._resource()
        if self._allow_without_attribute_name():
            if self._attribute is None:
                return
        attr = function.resolve(self._attribute)
        if attr not in res.attributes_schema:
            raise exception.InvalidTemplateAttribute(resource=self._resource_name, key=attr)

    def _result_ready(self, r):
        if r.action in (r.CREATE, r.ADOPT, r.SUSPEND, r.RESUME, r.UPDATE, r.ROLLBACK, r.SNAPSHOT, r.CHECK):
            return True
        return False

    def result(self):
        attr_name = function.resolve(self._attribute)
        resource = self._resource()
        if self._result_ready(resource):
            attribute = resource.FnGetAtt(attr_name)
        else:
            attribute = None
        if attribute is None:
            return None
        path_components = function.resolve(self._path_components)
        return attributes.select_from_attribute(attribute, path_components)