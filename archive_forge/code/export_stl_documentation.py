import math

    Yield the lines of an stl file corresponding to the solid given by face_dicts that is suitable for 3d printing.

    Arguments can be given to modify the model produced:

    - model='klein' -- (alt. 'poincare') the model of HH^3 to use.
    - cutout=False -- remove the interior of each face
    - shrink_factor=0.9 -- the fraction to cut out of each face
    - cuttoff_radius=0.9 -- maximum rescaling for projection into Poincaré model
    - num_subdivision=3 -- number of times to subdivide for the Poincaré model

    For printing domains in the Poincaré model, cutoff_radius is
    critical for avoiding infinitely thin cusps, which cannot be printed.
    