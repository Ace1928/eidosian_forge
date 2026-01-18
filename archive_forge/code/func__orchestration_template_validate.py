import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def _orchestration_template_validate(self, templ_name, parms):
    template_path = self.get_template_path(templ_name)
    cmd = 'orchestration template validate --template %s' % template_path
    for parm in parms:
        cmd += ' --parameter ' + parm
    ret = self.openstack(cmd)
    self.assertRegex(ret, 'Value:.*123')