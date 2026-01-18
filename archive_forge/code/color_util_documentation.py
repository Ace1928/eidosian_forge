from __future__ import annotations
from typing import Any, Callable, Collection, Tuple, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
Convert a color component (float or int) to a float from 0.0 to 1.0.

    Anything too small will become 0.0, and anything too large will become 1.0.
    