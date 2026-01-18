import cirq
import cirq_ft
import cirq_ft.infra.testing as cq_testing
import IPython.display
import ipywidgets
import pytest
from cirq_ft.infra.jupyter_tools import display_gate_and_compilation, svg_circuit
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def _mock_display(stuff):
    call_args.append(stuff)