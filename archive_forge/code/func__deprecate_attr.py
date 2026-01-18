from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def _deprecate_attr(self, attr, msg, version=None, date=None, collection_name=None, target=None, value=None, module=None):
    target, module, value_dict, trigger_dict = self._deprecate_setup(attr, target, module)
    value_dict[attr] = getattr(target, attr, value)
    trigger_dict[attr] = False

    def _trigger():
        if not trigger_dict[attr]:
            module.deprecate(msg, version=version, date=date, collection_name=collection_name)
            trigger_dict[attr] = True

    def _getter(_self):
        _trigger()
        return value_dict[attr]

    def _setter(_self, new_value):
        _trigger()
        value_dict[attr] = new_value
    prop = property(_getter)
    setattr(target, attr, prop)
    setattr(target, '_{0}_setter'.format(attr), prop.setter(_setter))