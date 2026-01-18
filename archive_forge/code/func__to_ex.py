from sympy.core import sympify, SympifyError
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public
def _to_ex(f, g):
    try:
        return f.__class__(g)
    except SympifyError:
        return None