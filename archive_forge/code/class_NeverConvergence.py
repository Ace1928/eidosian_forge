from ase.ga import get_raw_score
class NeverConvergence:
    """Test class that never converges."""

    def __init__(self):
        pass

    def converged(self):
        return False