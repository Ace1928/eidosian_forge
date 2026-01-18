import builtins
import os
import sys
import types

    Replaces the default __import__, to allow a module to be memorised
    before the user can change it
    