from .chemistry import equilibrium_quotient

    Solve equilibrium concentrations by using scipy.optimize.brentq

    Parameters
    ----------
    c0: array_like
        Initial guess of equilibrium concentrations
    stoich: tuple
        per specie stoichiometry coefficient (law of mass action)
    K: float
        equilibrium constant
    activity_product: callable
        see ``equilibrium_residual``
    delta_frac: float
        to avoid division by zero the span of searched values for
        the reactions coordinate (rc) is shrunk by 2*delta_frac*span(rc)
    