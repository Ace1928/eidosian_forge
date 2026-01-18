import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented

        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime
            comptime.breakpoint()

        And then, inside pdb, you can access 'ctx' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        