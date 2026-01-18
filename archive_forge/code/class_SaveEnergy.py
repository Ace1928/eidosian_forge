class SaveEnergy:
    """Class to save energy."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.energies = []

    def __call__(self):
        self.energies.append(atoms.get_total_energy())