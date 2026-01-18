important invariant is that the parts on the stack are themselves in
def _initialize_enumeration(self, multiplicities):
    """Allocates and initializes the partition stack.

        This is called from the enumeration/counting routines, so
        there is no need to call it separately."""
    num_components = len(multiplicities)
    cardinality = sum(multiplicities)
    self.pstack = [PartComponent() for i in range(num_components * cardinality + 1)]
    self.f = [0] * (cardinality + 1)
    for j in range(num_components):
        ps = self.pstack[j]
        ps.c = j
        ps.u = multiplicities[j]
        ps.v = multiplicities[j]
    self.f[0] = 0
    self.f[1] = num_components
    self.lpart = 0