import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def _template_validate(self, templ_name, parms):
    heat_template_path = self.get_template_path(templ_name)
    cmd = 'stack create test-stack --dry-run --template %s' % heat_template_path
    for parm in parms:
        cmd += ' --parameter ' + parm
    ret = self.openstack(cmd)
    self.assertRegex(ret, 'stack_name.*|.*test_stack')