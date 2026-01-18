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
class HOTemplate20141016(HOTemplate20130523):
    functions = {'get_attr': hot_funcs.GetAtt, 'get_file': hot_funcs.GetFile, 'get_param': hot_funcs.GetParam, 'get_resource': hot_funcs.GetResource, 'list_join': hot_funcs.Join, 'resource_facade': hot_funcs.ResourceFacade, 'str_replace': hot_funcs.Replace, 'Fn::Select': cfn_funcs.Select, 'Fn::GetAZs': hot_funcs.Removed, 'Fn::Join': hot_funcs.Removed, 'Fn::Split': hot_funcs.Removed, 'Fn::Replace': hot_funcs.Removed, 'Fn::Base64': hot_funcs.Removed, 'Fn::MemberListToMap': hot_funcs.Removed, 'Fn::ResourceFacade': hot_funcs.Removed, 'Ref': hot_funcs.Removed}