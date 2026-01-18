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