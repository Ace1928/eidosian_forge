import os
import numpy as np
import pennylane as qml
from pennylane.operation import active_new_opmath
def _import_of():
    """Import openfermion and openfermionpyscf."""
    try:
        import openfermion, openfermionpyscf
    except ImportError as Error:
        raise ImportError('This feature requires openfermionpyscf. It can be installed with: pip install openfermionpyscf.') from Error
    return (openfermion, openfermionpyscf)