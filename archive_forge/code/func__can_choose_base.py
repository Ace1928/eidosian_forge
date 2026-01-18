@staticmethod
def _can_choose_base(base, base_tree_remaining):
    for bases in base_tree_remaining:
        if not bases or bases[0] is base:
            continue
        for b in bases:
            if b is base:
                return False
    return True