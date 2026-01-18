from ase.io import read
import runpy
 Execute code on files.

    The given python code is evaluated on the Atoms object read from
    the input file for each frame of the file. Either of -e or -E
    option should provided for evaluating code given as a string or
    from a file, respectively.
    
    Variables which can be used inside the python code:
    - `index`: Index of the current Atoms object.
    - `atoms`: Current Atoms object.
    - `images`: List of all images given as input.
    