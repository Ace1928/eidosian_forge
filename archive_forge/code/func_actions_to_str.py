import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def actions_to_str(instructions):
    actions = []
    for instruction in instructions:
        if isinstance(instruction, ofproto_v1_2_parser.OFPInstructionActions):
            if instruction.type == ofproto_v1_2.OFPIT_APPLY_ACTIONS:
                for a in instruction.actions:
                    actions.append(action_to_str(a))
            elif instruction.type == ofproto_v1_2.OFPIT_WRITE_ACTIONS:
                write_actions = []
                for a in instruction.actions:
                    write_actions.append(action_to_str(a))
                if write_actions:
                    actions.append({'WRITE_ACTIONS': write_actions})
            elif instruction.type == ofproto_v1_2.OFPIT_CLEAR_ACTIONS:
                actions.append('CLEAR_ACTIONS')
            else:
                actions.append('UNKNOWN')
        elif isinstance(instruction, ofproto_v1_2_parser.OFPInstructionGotoTable):
            table_id = UTIL.ofp_table_to_user(instruction.table_id)
            buf = 'GOTO_TABLE:' + str(table_id)
            actions.append(buf)
        elif isinstance(instruction, ofproto_v1_2_parser.OFPInstructionWriteMetadata):
            buf = 'WRITE_METADATA:0x%x/0x%x' % (instruction.metadata, instruction.metadata_mask) if instruction.metadata_mask else 'WRITE_METADATA:0x%x' % instruction.metadata
            actions.append(buf)
        else:
            continue
    return actions