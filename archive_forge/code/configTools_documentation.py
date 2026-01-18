from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import (

        Get the value of an option. The value which is returned is the first
        provided among:

        1. a user-provided value in the options's ``self._values`` dict
        2. a caller-provided default value to this method call
        3. the global default for the option provided in ``fontTools.config``

        This is to provide the ability to migrate progressively from config
        options passed as arguments to fontTools APIs to config options read
        from the current TTFont, e.g.

        .. code:: python

            def fontToolsAPI(font, some_option):
                value = font.cfg.get("someLib.module:SOME_OPTION", some_option)
                # use value

        That way, the function will work the same for users of the API that
        still pass the option to the function call, but will favour the new
        config mechanism if the given font specifies a value for that option.
        