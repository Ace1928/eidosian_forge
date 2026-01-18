from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def _compare_aces(self, want, have, afi, name):
    """compares all aces"""

    def add_afi(entry, afi):
        """adds afi needed for
            setval processing"""
        if entry:
            entry['afi'] = afi
        return entry

    def pop_remark(r_entry, afi):
        """Takes out remarks from ace entry as remarks not same
            does not mean the ace entry to be re-introduced
            """
        if r_entry.get('remarks'):
            return r_entry.pop('remarks')
        else:
            return {}
    for wseq, wentry in iteritems(want):
        hentry = have.pop(wseq, {})
        rem_hentry, rem_wentry = ({}, {})
        if hentry:
            hentry = self.sanitize_protocol_options(wentry, hentry)
        if hentry != wentry:
            if hentry:
                rem_hentry['remarks'] = pop_remark(hentry, afi)
            if wentry:
                rem_wentry['remarks'] = pop_remark(wentry, afi)
            if hentry:
                if self.state == 'merged':
                    self._module.fail_json(msg='Cannot update existing sequence {0} of ACLs {1} with state merged. Please use state replaced or overridden.'.format(hentry.get('sequence', ''), name))
                else:
                    if rem_hentry.get('remarks'):
                        for k_hrems, hrems in rem_hentry.get('remarks').items():
                            if k_hrems not in rem_wentry.get('remarks', {}).keys():
                                if self.state in ['replaced', 'overridden']:
                                    self.addcmd({'remarks': hrems, 'sequence': hentry.get('sequence', '')}, 'remarks_no_data', negate=True)
                                    break
                                else:
                                    self.addcmd({'remarks': hrems, 'sequence': hentry.get('sequence', '')}, 'remarks', negate=True)
                    if hentry != wentry:
                        self.addcmd(add_afi(hentry, afi), 'aces', negate=True)
            if rem_wentry.get('remarks'):
                for k_wrems, wrems in rem_wentry.get('remarks').items():
                    if k_wrems not in rem_hentry.get('remarks', {}).keys():
                        self.addcmd({'remarks': wrems, 'sequence': hentry.get('sequence', '')}, 'remarks')
            if hentry != wentry:
                self.addcmd(add_afi(wentry, afi), 'aces')
    for hseq in have.values():
        if hseq.get('remarks'):
            for krems, rems in hseq.get('remarks').items():
                self.addcmd({'remarks': rems, 'sequence': hseq.get('sequence', '')}, 'remarks', negate=True)
        else:
            self.addcmd(add_afi(hseq, afi), 'aces', negate=True)