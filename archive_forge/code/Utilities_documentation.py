import warnings
from Bio import BiopythonDeprecationWarning
Print out a state sequence prediction in a nice manner.

    Arguments:
     - emissions -- The sequence of emissions of the sequence you are
       dealing with.
     - real_state -- The actual state path that generated the emissions.
     - predicted_state -- A state path predicted by some kind of HMM model.

    