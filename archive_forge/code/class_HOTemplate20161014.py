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
class HOTemplate20161014(HOTemplate20160408):
    CONDITIONS = 'conditions'
    SECTIONS = HOTemplate20160408.SECTIONS + (CONDITIONS,)
    SECTIONS_NO_DIRECT_ACCESS = HOTemplate20160408.SECTIONS_NO_DIRECT_ACCESS | set([CONDITIONS])
    _CFN_TO_HOT_SECTIONS = HOTemplate20160408._CFN_TO_HOT_SECTIONS
    _CFN_TO_HOT_SECTIONS.update({cfn_template.CfnTemplate.CONDITIONS: CONDITIONS})
    _EXTRA_RES_KEYS = RES_EXTERNAL_ID, RES_CONDITION = ('external_id', 'condition')
    _RESOURCE_KEYS = HOTemplate20160408._RESOURCE_KEYS + _EXTRA_RES_KEYS
    _RESOURCE_HOT_TO_CFN_ATTRS = HOTemplate20160408._RESOURCE_HOT_TO_CFN_ATTRS
    _RESOURCE_HOT_TO_CFN_ATTRS.update({RES_EXTERNAL_ID: None, RES_CONDITION: cfn_template.CfnTemplate.RES_CONDITION})
    OUTPUT_CONDITION = 'condition'
    OUTPUT_KEYS = HOTemplate20160408.OUTPUT_KEYS + (OUTPUT_CONDITION,)
    deletion_policies = {'Delete': rsrc_defn.ResourceDefinition.DELETE, 'Retain': rsrc_defn.ResourceDefinition.RETAIN, 'Snapshot': rsrc_defn.ResourceDefinition.SNAPSHOT, 'delete': rsrc_defn.ResourceDefinition.DELETE, 'retain': rsrc_defn.ResourceDefinition.RETAIN, 'snapshot': rsrc_defn.ResourceDefinition.SNAPSHOT}
    functions = {'get_attr': hot_funcs.GetAttAllAttributes, 'get_file': hot_funcs.GetFile, 'get_param': hot_funcs.GetParam, 'get_resource': hot_funcs.GetResource, 'list_join': hot_funcs.JoinMultiple, 'repeat': hot_funcs.RepeatWithMap, 'resource_facade': hot_funcs.ResourceFacade, 'str_replace': hot_funcs.ReplaceJson, 'digest': hot_funcs.Digest, 'str_split': hot_funcs.StrSplit, 'map_merge': hot_funcs.MapMerge, 'yaql': hot_funcs.Yaql, 'map_replace': hot_funcs.MapReplace, 'if': hot_funcs.If, 'Fn::Select': hot_funcs.Removed, 'Fn::GetAZs': hot_funcs.Removed, 'Fn::Join': hot_funcs.Removed, 'Fn::Split': hot_funcs.Removed, 'Fn::Replace': hot_funcs.Removed, 'Fn::Base64': hot_funcs.Removed, 'Fn::MemberListToMap': hot_funcs.Removed, 'Fn::ResourceFacade': hot_funcs.Removed, 'Ref': hot_funcs.Removed}
    condition_functions = {'get_param': hot_funcs.GetParam, 'equals': hot_funcs.Equals, 'not': hot_funcs.Not, 'and': hot_funcs.And, 'or': hot_funcs.Or}

    def __init__(self, tmpl, template_id=None, files=None, env=None):
        super(HOTemplate20161014, self).__init__(tmpl, template_id, files, env)
        self._parser_condition_functions = {}
        for n, f in self.functions.items():
            if not f == hot_funcs.Removed:
                self._parser_condition_functions[n] = function.Invalid
            else:
                self._parser_condition_functions[n] = f
        self._parser_condition_functions.update(self.condition_functions)
        self.merge_sections = [self.PARAMETERS, self.CONDITIONS]

    def _get_condition_definitions(self):
        return self.t.get(self.CONDITIONS, {})

    def _rsrc_defn_args(self, stack, name, data):
        for arg in super(HOTemplate20161014, self)._rsrc_defn_args(stack, name, data):
            yield arg
        parse = functools.partial(self.parse, stack)
        parse_cond = functools.partial(self.parse_condition, stack)
        yield ('external_id', self._parse_resource_field(self.RES_EXTERNAL_ID, (str, function.Function), 'string', name, data, parse))
        yield ('condition', self._parse_resource_field(self.RES_CONDITION, (str, bool, function.Function), 'string_or_boolean', name, data, parse_cond))