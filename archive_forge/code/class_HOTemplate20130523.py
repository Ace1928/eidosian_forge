import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
class HOTemplate20130523(template_common.CommonTemplate):
    """A Heat Orchestration Template format stack template."""
    SECTIONS = VERSION, DESCRIPTION, PARAMETER_GROUPS, PARAMETERS, RESOURCES, OUTPUTS, MAPPINGS = ('heat_template_version', 'description', 'parameter_groups', 'parameters', 'resources', 'outputs', '__undefined__')
    OUTPUT_KEYS = OUTPUT_DESCRIPTION, OUTPUT_VALUE = ('description', 'value')
    SECTIONS_NO_DIRECT_ACCESS = set([PARAMETERS, VERSION])
    _CFN_TO_HOT_SECTIONS = {cfn_template.CfnTemplate.VERSION: VERSION, cfn_template.CfnTemplate.DESCRIPTION: DESCRIPTION, cfn_template.CfnTemplate.PARAMETERS: PARAMETERS, cfn_template.CfnTemplate.MAPPINGS: MAPPINGS, cfn_template.CfnTemplate.RESOURCES: RESOURCES, cfn_template.CfnTemplate.OUTPUTS: OUTPUTS}
    _RESOURCE_KEYS = RES_TYPE, RES_PROPERTIES, RES_METADATA, RES_DEPENDS_ON, RES_DELETION_POLICY, RES_UPDATE_POLICY, RES_DESCRIPTION = ('type', 'properties', 'metadata', 'depends_on', 'deletion_policy', 'update_policy', 'description')
    _RESOURCE_HOT_TO_CFN_ATTRS = {RES_TYPE: cfn_template.CfnTemplate.RES_TYPE, RES_PROPERTIES: cfn_template.CfnTemplate.RES_PROPERTIES, RES_METADATA: cfn_template.CfnTemplate.RES_METADATA, RES_DEPENDS_ON: cfn_template.CfnTemplate.RES_DEPENDS_ON, RES_DELETION_POLICY: cfn_template.CfnTemplate.RES_DELETION_POLICY, RES_UPDATE_POLICY: cfn_template.CfnTemplate.RES_UPDATE_POLICY, RES_DESCRIPTION: cfn_template.CfnTemplate.RES_DESCRIPTION}
    _HOT_TO_CFN_ATTRS = _RESOURCE_HOT_TO_CFN_ATTRS
    _HOT_TO_CFN_ATTRS.update({OUTPUT_VALUE: cfn_template.CfnTemplate.OUTPUT_VALUE})
    functions = {'Fn::GetAZs': cfn_funcs.GetAZs, 'get_param': hot_funcs.GetParam, 'get_resource': hot_funcs.GetResource, 'Ref': cfn_funcs.Ref, 'get_attr': hot_funcs.GetAttThenSelect, 'Fn::Select': cfn_funcs.Select, 'Fn::Join': cfn_funcs.Join, 'list_join': hot_funcs.Join, 'Fn::Split': cfn_funcs.Split, 'str_replace': hot_funcs.Replace, 'Fn::Replace': cfn_funcs.Replace, 'Fn::Base64': cfn_funcs.Base64, 'Fn::MemberListToMap': cfn_funcs.MemberListToMap, 'resource_facade': hot_funcs.ResourceFacade, 'Fn::ResourceFacade': cfn_funcs.ResourceFacade, 'get_file': hot_funcs.GetFile}
    deletion_policies = {'Delete': rsrc_defn.ResourceDefinition.DELETE, 'Retain': rsrc_defn.ResourceDefinition.RETAIN, 'Snapshot': rsrc_defn.ResourceDefinition.SNAPSHOT}
    param_schema_class = parameters.HOTParamSchema

    def __getitem__(self, section):
        """"Get the relevant section in the template."""
        if section not in self.SECTIONS:
            section = HOTemplate20130523._translate(section, self._CFN_TO_HOT_SECTIONS, _('"%s" is not a valid template section'))
        if section not in self.SECTIONS:
            raise KeyError(_('"%s" is not a valid template section') % section)
        if section in self.SECTIONS_NO_DIRECT_ACCESS:
            raise KeyError(_('Section %s can not be accessed directly.') % section)
        if section == self.MAPPINGS:
            return {}
        if section == self.DESCRIPTION:
            default = 'No description'
        else:
            default = {}
        the_section = self.t.get(section) or default
        if section == self.RESOURCES:
            return self._translate_resources(the_section)
        if section == self.OUTPUTS:
            self.validate_section(self.OUTPUTS, self.OUTPUT_VALUE, the_section, self.OUTPUT_KEYS)
        return the_section

    @staticmethod
    def _translate(value, mapping, err_msg=None):
        try:
            return mapping[value]
        except KeyError as ke:
            if err_msg:
                raise KeyError(err_msg % value)
            else:
                raise ke

    def validate_section(self, section, sub_section, data, allowed_keys):
        obj_name = section[:-1]
        err_msg = _('"%%s" is not a valid keyword inside a %s definition') % obj_name
        args = {'object_name': obj_name, 'sub_section': sub_section}
        message = _('Each %(object_name)s must contain a %(sub_section)s key.') % args
        for name, attrs in sorted(data.items()):
            if not attrs:
                raise exception.StackValidationFailed(message=message)
            try:
                for attr, attr_value in attrs.items():
                    if attr not in allowed_keys:
                        raise KeyError(err_msg % attr)
                if sub_section not in attrs:
                    raise exception.StackValidationFailed(message=message)
            except AttributeError:
                message = _('"%(section)s" must contain a map of %(obj_name)s maps. Found a [%(_type)s] instead') % {'section': section, '_type': type(attrs), 'obj_name': obj_name}
                raise exception.StackValidationFailed(message=message)
            except KeyError as e:
                raise exception.StackValidationFailed(message=e.args[0])

    def _translate_section(self, section, sub_section, data, mapping):
        self.validate_section(section, sub_section, data, mapping)
        cfn_objects = {}
        for name, attrs in sorted(data.items()):
            cfn_object = {}
            for attr, attr_value in attrs.items():
                cfn_attr = mapping[attr]
                if cfn_attr is not None:
                    cfn_object[cfn_attr] = attr_value
            cfn_objects[name] = cfn_object
        return cfn_objects

    def _translate_resources(self, resources):
        """Get the resources of the template translated into CFN format."""
        return self._translate_section(self.RESOURCES, self.RES_TYPE, resources, self._RESOURCE_HOT_TO_CFN_ATTRS)

    def get_section_name(self, section):
        cfn_to_hot_attrs = dict(zip(self._HOT_TO_CFN_ATTRS.values(), self._HOT_TO_CFN_ATTRS.keys()))
        return cfn_to_hot_attrs.get(section, section)

    def param_schemata(self, param_defaults=None):
        parameter_section = self.t.get(self.PARAMETERS) or {}
        pdefaults = param_defaults or {}
        for name, schema in parameter_section.items():
            if name in pdefaults:
                parameter_section[name]['default'] = pdefaults[name]
        params = parameter_section.items()
        return dict(((name, self.param_schema_class.from_dict(name, schema)) for name, schema in params))

    def parameters(self, stack_identifier, user_params, param_defaults=None):
        return parameters.HOTParameters(stack_identifier, self, user_params=user_params, param_defaults=param_defaults)

    def resource_definitions(self, stack):
        resources = self.t.get(self.RESOURCES) or {}
        conditions = self.conditions(stack)
        valid_keys = frozenset(self._RESOURCE_KEYS)

        def defns():
            for name, snippet in resources.items():
                try:
                    invalid_keys = set(snippet) - valid_keys
                    if invalid_keys:
                        raise ValueError(_('Invalid keyword(s) inside a resource definition: %s') % ', '.join(invalid_keys))
                    defn_data = dict(self._rsrc_defn_args(stack, name, snippet))
                except (TypeError, ValueError, KeyError) as ex:
                    msg = str(ex)
                    raise exception.StackValidationFailed(message=msg)
                defn = rsrc_defn.ResourceDefinition(name, **defn_data)
                cond_name = defn.condition()
                if cond_name is not None:
                    try:
                        enabled = conditions.is_enabled(cond_name)
                    except ValueError as exc:
                        path = [self.RESOURCES, name, self.RES_CONDITION]
                        message = str(exc)
                        raise exception.StackValidationFailed(path=path, message=message)
                    if not enabled:
                        continue
                yield (name, defn)
        return dict(defns())

    def add_resource(self, definition, name=None):
        if name is None:
            name = definition.name
        if self.t.get(self.RESOURCES) is None:
            self.t[self.RESOURCES] = {}
        rendered = definition.render_hot()
        dep_list = rendered.get(self.RES_DEPENDS_ON)
        if dep_list:
            rendered[self.RES_DEPENDS_ON] = [d for d in dep_list if d in self.t[self.RESOURCES]]
        self.t[self.RESOURCES][name] = rendered

    def add_output(self, definition):
        if self.t.get(self.OUTPUTS) is None:
            self.t[self.OUTPUTS] = {}
        self.t[self.OUTPUTS][definition.name] = definition.render_hot()