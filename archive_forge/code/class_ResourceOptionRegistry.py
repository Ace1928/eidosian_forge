from keystone.common import validation
from keystone.i18n import _
class ResourceOptionRegistry(object):

    def __init__(self, registry_name):
        self._registered_options = {}
        self._registry_type = registry_name

    @property
    def option_names(self):
        return set([opt.option_name for opt in self.options])

    @property
    def options_by_name(self):
        return {opt.option_name: opt for opt in self._registered_options.values()}

    @property
    def options(self):
        return self._registered_options.values()

    @property
    def option_ids(self):
        return set(self._registered_options.keys())

    def get_option_by_id(self, opt_id):
        return self._registered_options.get(opt_id, None)

    def get_option_by_name(self, name):
        for option in self._registered_options.values():
            if name == option.option_name:
                return option
        return None

    @property
    def json_schema(self):
        schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
        for opt in self.options:
            if opt.json_schema is not None:
                schema['properties'][opt.option_name] = validation.nullable(opt.json_schema)
            else:
                schema['properties'][opt.option_name] = {}
        return schema

    def register_option(self, option):
        if option in self.options:
            return
        if option.option_id in self._registered_options:
            raise ValueError(_('Option %(option_id)s already defined in %(registry)s.') % {'option_id': option.option_id, 'registry': self._registry_type})
        if option.option_name in self.option_names:
            raise ValueError(_('Option %(option_name)s already defined in %(registry)s') % {'option_name': option.option_name, 'registry': self._registry_type})
        self._registered_options[option.option_id] = option