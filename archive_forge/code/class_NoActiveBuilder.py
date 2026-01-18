from qiskit.exceptions import QiskitError
class NoActiveBuilder(PulseError):
    """Raised if no builder context is active."""