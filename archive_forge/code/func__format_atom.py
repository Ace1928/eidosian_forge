import abc
from taskflow.persistence import models
def _format_atom(atom_detail):
    return {'atom': atom_detail.to_dict(), 'type': models.atom_detail_type(atom_detail)}