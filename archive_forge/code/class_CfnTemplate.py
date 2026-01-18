import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_params
from heat.engine import function
from heat.engine import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
class CfnTemplate(CfnTemplateBase):
    CONDITIONS = 'Conditions'
    SECTIONS = CfnTemplateBase.SECTIONS + (CONDITIONS,)
    SECTIONS_NO_DIRECT_ACCESS = CfnTemplateBase.SECTIONS_NO_DIRECT_ACCESS | set([CONDITIONS])
    RES_CONDITION = 'Condition'
    _RESOURCE_KEYS = CfnTemplateBase._RESOURCE_KEYS + (RES_CONDITION,)
    HOT_TO_CFN_RES_ATTRS = CfnTemplateBase.HOT_TO_CFN_RES_ATTRS
    HOT_TO_CFN_RES_ATTRS.update({'condition': RES_CONDITION})
    OUTPUT_CONDITION = 'Condition'
    OUTPUT_KEYS = CfnTemplateBase.OUTPUT_KEYS + (OUTPUT_CONDITION,)
    functions = {'Fn::FindInMap': cfn_funcs.FindInMap, 'Fn::GetAZs': cfn_funcs.GetAZs, 'Ref': cfn_funcs.Ref, 'Fn::GetAtt': cfn_funcs.GetAtt, 'Fn::Select': cfn_funcs.Select, 'Fn::Join': cfn_funcs.Join, 'Fn::Split': cfn_funcs.Split, 'Fn::Replace': cfn_funcs.Replace, 'Fn::Base64': cfn_funcs.Base64, 'Fn::MemberListToMap': cfn_funcs.MemberListToMap, 'Fn::ResourceFacade': cfn_funcs.ResourceFacade, 'Fn::If': cfn_funcs.If}
    condition_functions = {'Fn::Equals': cfn_funcs.Equals, 'Ref': cfn_funcs.ParamRef, 'Fn::FindInMap': cfn_funcs.FindInMap, 'Fn::Not': cfn_funcs.Not, 'Fn::And': cfn_funcs.And, 'Fn::Or': cfn_funcs.Or}

    def __init__(self, tmpl, template_id=None, files=None, env=None):
        super(CfnTemplate, self).__init__(tmpl, template_id, files, env)
        self.merge_sections = [self.PARAMETERS, self.CONDITIONS]

    def _get_condition_definitions(self):
        return self.t.get(self.CONDITIONS, {})

    def _rsrc_defn_args(self, stack, name, data):
        for arg in super(CfnTemplate, self)._rsrc_defn_args(stack, name, data):
            yield arg
        parse_cond = functools.partial(self.parse_condition, stack)
        yield ('condition', self._parse_resource_field(self.RES_CONDITION, (str, bool, function.Function), 'string or boolean', name, data, parse_cond))