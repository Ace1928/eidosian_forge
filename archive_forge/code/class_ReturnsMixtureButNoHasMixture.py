import pytest
import numpy as np
import cirq
class ReturnsMixtureButNoHasMixture:

    def _mixture_(self):
        return ((0.4, 'a'), (0.6, 'b'))