from ..common.utils import bytes2str
def DIE_is_ptr_to_member_struct(type_die):
    if type_die.tag == 'DW_TAG_structure_type':
        members = tuple((die for die in type_die.iter_children() if die.tag == 'DW_TAG_member'))
        return len(members) == 2 and safe_DIE_name(members[0]) == '__pfn' and (safe_DIE_name(members[1]) == '__delta')
    return False