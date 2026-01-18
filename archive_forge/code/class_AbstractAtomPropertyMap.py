class AbstractAtomPropertyMap(AbstractPropertyMap):
    """Define class for atom properties map."""

    def __init__(self, property_dict, property_keys, property_list):
        """Initialize the class."""
        AbstractPropertyMap.__init__(self, property_dict, property_keys, property_list)

    def _translate_id(self, ent_id):
        """Return entity identifier on atoms (PRIVATE)."""
        if len(ent_id) == 4:
            chain_id, res_id, atom_name, icode = ent_id
        else:
            chain_id, res_id, atom_name = ent_id
            icode = None
        if isinstance(res_id, int):
            ent_id = (chain_id, (' ', res_id, ' '), atom_name, icode)
        return ent_id