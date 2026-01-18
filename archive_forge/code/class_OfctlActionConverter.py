import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
class OfctlActionConverter(object):

    @classmethod
    def goto_table(cls, ofproto, action_str):
        assert action_str.startswith('goto_table:')
        table_id = str_to_int(action_str[len('goto_table:'):])
        return dict(OFPInstructionGotoTable={'table_id': table_id})

    @classmethod
    def normal(cls, ofproto, action_str):
        return cls.output(ofproto, action_str)

    @classmethod
    def output(cls, ofproto, action_str):
        if action_str == 'normal':
            port = ofproto.OFPP_NORMAL
        else:
            assert action_str.startswith('output:')
            port = str_to_int(action_str[len('output:'):])
        return dict(OFPActionOutput={'port': port})

    @classmethod
    def pop_vlan(cls, ofproto, action_str):
        return dict(OFPActionPopVlan={})

    @classmethod
    def set_field(cls, ofproto, action_str):
        try:
            assert action_str.startswith('set_field:')
            value, key = action_str[len('set_field:'):].split('->', 1)
            fieldarg = dict(field=ofp_ofctl_field_name_to_os_ken(key))
            m = value.find('/')
            if m >= 0:
                fieldarg['value'] = str_to_int(value[:m])
                fieldarg['mask'] = str_to_int(value[m + 1:])
            else:
                fieldarg['value'] = str_to_int(value)
        except Exception:
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
        return dict(OFPActionSetField={'field': {'OXMTlv': fieldarg}})

    @classmethod
    def resubmit(cls, ofproto, action_str):
        arg = action_str[len('resubmit'):]
        kwargs = {}
        try:
            if arg[0] == ':':
                kwargs['in_port'] = str_to_int(arg[1:])
            elif arg[0] == '(' and arg[-1] == ')':
                in_port, table_id = arg[1:-1].split(',')
                if in_port:
                    kwargs['in_port'] = str_to_int(in_port)
                if table_id:
                    kwargs['table_id'] = str_to_int(table_id)
            else:
                raise Exception
            return dict(NXActionResubmitTable=kwargs)
        except Exception:
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)

    @classmethod
    def conjunction(cls, ofproto, action_str):
        try:
            assert action_str.startswith('conjunction(')
            assert action_str[-1] == ')'
            args = action_str[len('conjunction('):-1].split(',')
            assert len(args) == 2
            id_ = str_to_int(args[0])
            clauses = list(map(str_to_int, args[1].split('/')))
            assert len(clauses) == 2
            return dict(NXActionConjunction={'clause': clauses[0] - 1, 'n_clauses': clauses[1], 'id': id_})
        except Exception:
            raise os_ken.exception.OFPInvalidActionString(action_str=action_str)

    @classmethod
    def ct(cls, ofproto, action_str):
        str_to_port = {'ftp': 21, 'tftp': 69}
        flags = 0
        zone_src = ''
        zone_ofs_nbits = 0
        recirc_table = nicira_ext.NX_CT_RECIRC_NONE
        alg = 0
        ct_actions = []
        if len(action_str) > 2:
            if not action_str.startswith('ct(') or action_str[-1] != ')':
                raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
            rest = tokenize_ofp_instruction_arg(action_str[len('ct('):-1])
        else:
            rest = []
        for arg in rest:
            if arg == 'commit':
                flags |= nicira_ext.NX_CT_F_COMMIT
                rest = rest[len('commit'):]
            elif arg == 'force':
                flags |= nicira_ext.NX_CT_F_FORCE
            elif arg.startswith('exec('):
                ct_actions = ofp_instruction_from_str(ofproto, arg[len('exec('):-1])
            else:
                try:
                    k, v = arg.split('=', 1)
                    if k == 'table':
                        recirc_table = str_to_int(v)
                    elif k == 'zone':
                        m = re.search('\\[(\\d*)\\.\\.(\\d*)\\]', v)
                        if m:
                            zone_ofs_nbits = nicira_ext.ofs_nbits(int(m.group(1)), int(m.group(2)))
                            zone_src = nxm_field_name_to_os_ken(v[:m.start(0)])
                        else:
                            zone_ofs_nbits = str_to_int(v)
                    elif k == 'alg':
                        alg = str_to_port[arg[len('alg='):]]
                except Exception:
                    raise os_ken.exception.OFPInvalidActionString(action_str=action_str)
        return dict(NXActionCT={'flags': flags, 'zone_src': zone_src, 'zone_ofs_nbits': zone_ofs_nbits, 'recirc_table': recirc_table, 'alg': alg, 'actions': ct_actions})

    @classmethod
    def ct_clear(cls, ofproto, action_str):
        return dict(NXActionCTClear={})