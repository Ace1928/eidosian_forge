from . import _ccallback_c
import ctypes

        Create a low-level callback function from an exported Cython function.

        Parameters
        ----------
        module : module
            Cython module where the exported function resides
        name : str
            Name of the exported function
        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
            User data to pass on to the callback function.
        signature : str, optional
            Signature of the function. If omitted, determined from *function*.

        