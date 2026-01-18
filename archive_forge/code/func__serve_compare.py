from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.ntp_global import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _serve_compare(self, want, have):
    serve_want = want.pop('serve', {})
    serve_have = have.pop('serve', {})
    for name, entry in iteritems(serve_want):
        if name == 'all' and entry:
            w = {'serve': {'all': True}}
            self.compare(parsers='serve_all', want=w, have={'serve': {'all': serve_have.pop('all', False)}})
        else:
            for k_afi, v_afi in iteritems(entry):
                for k, v in iteritems(v_afi):
                    afi = v_afi['afi']
                    if k == 'afi':
                        continue
                    h = {}
                    if k == 'acls':
                        for ace, ace_entry in iteritems(v):
                            if serve_have.get('access_lists'):
                                for hk, hv in iteritems(serve_have['access_lists']):
                                    for h_k, h_v in iteritems(hv):
                                        h_afi = hv['afi']
                                        if h_k == 'afi':
                                            continue
                                        if h_afi == afi:
                                            if ace in h_v:
                                                h_acc = {'afi': h_afi, 'acls': h_v.pop(ace)}
                                                h = {'serve': {'access_lists': h_acc}}
                            w = {'serve': {'access_lists': {'afi': afi, 'acls': ace_entry}}}
                            self.compare(parsers='serve', want=w, have=h)
    for k, v in iteritems(serve_have):
        if k == 'all' and v:
            h = {'serve': {'all': True}}
            self.compare(parsers='serve_all', want={}, have=h)
        else:
            for k_afi, v_afi in iteritems(v):
                for k, v in iteritems(v_afi):
                    hafi = v_afi['afi']
                    if k == 'afi':
                        continue
                    for k_acl, v_acl in iteritems(v):
                        h = {'serve': {'access_lists': {'afi': hafi, 'acls': v_acl}}}
                        self.compare(parsers='serve', want={}, have=h)