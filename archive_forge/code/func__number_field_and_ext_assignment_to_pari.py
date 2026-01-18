from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def _number_field_and_ext_assignment_to_pari(number_field, ext_assignment):
    if number_field:
        pari_number_field = pari(number_field)
    else:
        pari_number_field = None

    def item_to_pari(item):
        key, value = item
        if pari_number_field:
            return (key, pari(value).Mod(pari_number_field))
        else:
            return (key, pari(value))
    pari_ext_assignment = dict([item_to_pari(item) for item in ext_assignment.items()])
    return (pari_number_field, pari_ext_assignment)