import os
import subprocess
Run a paml program using the current configuration.

        Check that the class attributes exist and raise an error
        if not. Then run the command and check if it succeeds with
        a return code of 0, otherwise raise an error.

        The arguments may be passed as either absolute or relative
        paths, despite the fact that paml requires relative paths.
        