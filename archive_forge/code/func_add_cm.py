from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def add_cm(self, *args: Any) -> TextStateManagerChainMapType:
    """Concatenate an additional transform matrix"""
    self.transform_stack = self.reset_tm()
    self.q_queue.update(self.q_depth[-1:])
    self.transform_stack = self.transform_stack.new_child(self.new_transform(*args))
    return self.transform_stack